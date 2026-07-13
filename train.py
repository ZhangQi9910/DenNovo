# Canonical DenNovo model, preprocessing, losses, and training workflow.

# This section imports OS utilities and sets TensorFlow GPU/runtime environment variables before importing TensorFlow.
import argparse
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# This section imports TensorFlow and applies global runtime options.
import tensorflow as tf
tf.config.optimizer.set_jit(False)

# This section enables GPU memory growth as a second-level safeguard.
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Detected {len(gpus)} GPU(s), memory growth is enabled.")
    except RuntimeError as e:
        print("GPU configuration failed because GPUs were already initialized.")
        print("Please restart the kernel/runtime and run again.")
        print(e)

# This section imports general-purpose data, plotting, numeric, and utility libraries.
import numpy as np
import numba as nb
import random
import math
from dataclasses import dataclass, asdict
import gc

# This section imports proteomics-specific MGF and mass calculation utilities.
from pyteomics import mgf, mass

print(f"TensorFlow Version: {tf.__version__}")

# This section loads either standalone Keras or the TensorFlow-bundled API.
try:
    import keras as k
except ImportError:
    from tensorflow import keras as k

K = k.backend
Layer = k.layers.Layer
Model = k.Model
Input = k.Input
Conv1D = k.layers.Conv1D
Dense = k.layers.Dense
Add = k.layers.Add
Activation = k.layers.Activation
BatchNormalization = k.layers.BatchNormalization
LayerNormalization = k.layers.LayerNormalization
GlobalAveragePooling1D = k.layers.GlobalAveragePooling1D
Dropout = k.layers.Dropout
SpatialDropout1D = k.layers.SpatialDropout1D
ModelCheckpoint = k.callbacks.ModelCheckpoint
ReduceLROnPlateau = k.callbacks.ReduceLROnPlateau
EarlyStopping = k.callbacks.EarlyStopping
CSVLogger = k.callbacks.CSVLogger
categorical_crossentropy = k.losses.categorical_crossentropy
binary_crossentropy = k.losses.binary_crossentropy
MSE = k.losses.MeanSquaredError()
sparse_categorical_crossentropy = k.losses.sparse_categorical_crossentropy
Adam = k.optimizers.Adam
regularizers = k.regularizers

if hasattr(k.layers, 'GroupNormalization'):
    GroupNormalization = k.layers.GroupNormalization
else:
    from tensorflow_addons.layers import GroupNormalization

# This section clears previous TensorFlow/Keras sessions to avoid stale graph memory.
K.clear_session()
gc.collect()
print()
print(f"GPU available: {bool(tf.config.list_physical_devices('GPU'))}")

# This section patches missing backend functions for Keras 3.x compatibility with legacy code paths.
_tf_funcs = [
    'shape', 'reshape', 'transpose', 'expand_dims', 'squeeze',
    'tile', 'gather', 'one_hot', 'reverse',
    'cast',
    'argmax', 'argmin',
    'abs', 'square', 'sqrt', 'exp', 'pow', 'sign', 'round', 'floor',
    'equal', 'not_equal', 'greater', 'greater_equal', 'less', 'less_equal',
    'where',
    'zeros', 'ones', 'zeros_like', 'ones_like',
    'stop_gradient', 'identity',
]
_tf_reduce_funcs = {
    'sum': 'reduce_sum',
    'mean': 'reduce_mean',
    'max': 'reduce_max',
    'min': 'reduce_min',
    'prod': 'reduce_prod',
    'any': 'reduce_any',
    'all': 'reduce_all',
}
_tf_nn_funcs = ['sigmoid', 'softmax', 'relu', 'tanh', 'softplus']
_tf_math_funcs = ['log', 'ceil']

for name in _tf_funcs:
    if not hasattr(K, name) and hasattr(tf, name):
        setattr(K, name, getattr(tf, name))

for k_name, tf_name in _tf_reduce_funcs.items():
    if not hasattr(K, k_name) and hasattr(tf, tf_name):
        setattr(K, k_name, getattr(tf, tf_name))

for name in _tf_nn_funcs:
    if not hasattr(K, name) and hasattr(tf.nn, name):
        setattr(K, name, getattr(tf.nn, name))

for name in _tf_math_funcs:
    if not hasattr(K, name) and hasattr(tf.math, name):
        setattr(K, name, getattr(tf.math, name))

if not hasattr(K, 'ndim'):
    K.ndim = lambda x: len(x.shape)
if not hasattr(K, 'int_shape'):
    K.int_shape = lambda x: tuple(x.shape.as_list()) if hasattr(x.shape, 'as_list') else tuple(x.shape)
if not hasattr(K, 'batch_flatten'):
    K.batch_flatten = lambda x: tf.reshape(x, [tf.shape(x)[0], -1])
if not hasattr(K, 'clip'):
    K.clip = tf.clip_by_value
if not hasattr(K, 'maximum'):
    K.maximum = tf.maximum
if not hasattr(K, 'minimum'):
    K.minimum = tf.minimum
if not hasattr(K, 'concatenate'):
    K.concatenate = tf.concat
if not hasattr(K, 'stack'):
    K.stack = tf.stack
if not hasattr(K, 'flatten'):
    K.flatten = lambda x: tf.reshape(x, [-1])
if not hasattr(K, 'repeat_elements'):
    K.repeat_elements = lambda x, rep, axis: tf.repeat(x, rep, axis=axis)
if not hasattr(K, 'batch_dot'):
    K.batch_dot = lambda x, y, axes=None: tf.linalg.matmul(x, y)
if not hasattr(K, 'dot'):
    K.dot = lambda x, y: tf.tensordot(x, y, axes=1)
if not hasattr(K, 'switch'):
    K.switch = tf.where
if not hasattr(K, 'random_uniform'):
    K.random_uniform = tf.random.uniform
if not hasattr(K, 'random_normal'):
    K.random_normal = tf.random.normal
if not hasattr(K, 'epsilon'):
    K.epsilon = lambda: 1e-7
if not hasattr(K, 'floatx'):
    K.floatx = lambda: 'float32'

print("Keras 3.x compatibility patch loaded.")

# This section defines the diffusion refinement module used to denoise sequence logits.
class DiffusionModel(Model):
    def __init__(self, seq_length, channels):
        super().__init__()
        self.seq_length = seq_length
        self.channels = channels
        self.denoise_blocks = [
            Conv1D(channels * 2, 5, padding='same', activation='gelu'),
            GroupNormalization(groups=-1),
            Conv1D(channels, 3, padding='same', activation='gelu'),
            GroupNormalization(groups=-1)
        ]
    def denoise(self, x):
        for layer in self.denoise_blocks:
            x = layer(x)
        return x

    def call(self, inputs, training=None):
        if training:
            noise_mask = tf.random.normal(tf.shape(inputs)) * (1.0 - tf.reduce_max(inputs, axis=-1, keepdims=True))
            inputs = inputs + noise_mask
        return self.denoise(inputs)

# This section defines utility casting/helpers and a config wrapper.
def asnp(x): return np.asarray(x)
def asnp32(x): return np.asarray(x, dtype='float32')
def np32(x): return np.array(x, dtype='float32')
def zero32(shape): return np.zeros(shape, dtype='float32')
def clipn(*kw, sigma=4): return np.clip(np.random.randn(*kw), -sigma, sigma) / sigma

class config(dict):
    def __init__(self, *args, **kwargs):
        super(config, self).__init__(*args, **kwargs)
        self.__dict__ = self

# This section defines a robust Sequence data loader for train/validation/test generators.
class data_seq(k.utils.Sequence):
    def __init__(self, sps, processor, batch_size, shuffle=1, xonly=1, **kws):
        self.sps = list(sps)
        self.processor = processor
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.xonly = xonly
        self.kws = kws
        if self.shuffle:
            np.random.shuffle(self.sps)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.sps)

    def __len__(self):
        return math.ceil(len(self.sps) / self.batch_size)

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min(start_idx + self.batch_size, len(self.sps))
        batch_sps = self.sps[start_idx:end_idx]
        if len(batch_sps) == 0:
            raise ValueError(f"Empty batch at index {idx}")

        result = self.processor(batch_sps, **self.kws)
        if self.xonly:
            if isinstance(result, tuple) and len(result) == 2 and isinstance(result[1], dict):
                return result[0]
            return result

        inputs, labels = result
        for key, value in labels.items():
            if key == 'charge':
                labels[key] = np.array(value, dtype=np.int32)
        return result

# This section defines mass/ppm helper functions for spectrum validation.
def fastmass(pep, ion_type, charge, nmod=None, mod=None, cam=True):
    if ion_type == 'M' and all(aa in mono for aa in pep):
        water_mass = 18.010565
        proton_mass = 1.00727647
        neutral_mass = sum(mono[aa] for aa in pep) + water_mass
        return (neutral_mass + charge * proton_mass) / charge
    base = mass.fast_mass(pep, ion_type=ion_type, charge=charge)
    if cam:
        base += 57.021 * pep.count('C') / charge
    return base

def m1(pep, c=1, **kws): return fastmass(pep, ion_type='M', charge=c, **kws)
def ppm(m1, m2): return ((m1 - m2) / m1) * 1000000

def ppmdiff(sp, pep=None):
    if pep is None:
        pep = sp['pep']
    mass_val = fastmass(pep, 'M', sp['charge'], mod=sp['mod'], nmod=sp['nmod'])
    return ((sp['mass'] - mass_val) / mass_val) * 1000000

# This section defines numba-accelerated intensity normalization and vectorization.
@nb.njit
def normalize(it, mode):
    if mode == 0:
        return it
    elif mode == 2:
        return np.sqrt(it)
    elif mode == 3:
        return np.sqrt(np.sqrt(it))
    elif mode == 4:
        return np.sqrt(np.sqrt(np.sqrt(it)))
    return it

@nb.njit
def remove_precursor(v, pre_mz, c, precision, low, r):
    for delta in (0, 1, 2):
        mz = pre_mz + delta / c
        if mz > 0 and mz >= low:
            pc = round((mz - low) / precision)
            if pc - r < len(v):
                v[max(0, pc - r):min(len(v), pc + r)] = 0
    return None

def kth(v, k):
    return np.partition(v, k)[k]

@nb.njit
def mz2pos(mzs, pre, low): return round(mzs / pre + low)

@nb.njit
def flat(v, mz, it, pre, low, use_max):
    for i, x in enumerate(mz):
        pos = int(round((x - low) / pre))
        if pos < 0 or pos >= len(v):
            continue
        if use_max:
            v[pos] = max(v[pos], it[i])
        else:
            v[pos] += it[i]
    return v

@nb.njit
def native_vectorlize(mz, it, mass_val, c, precision, dim, low, mode, v, kth_val, th, de, dn, use_max):
    max_it = 0.0
    if it.size > 0:
        max_it = np.max(it)
    if max_it > 0.0:
        it /= max_it

    if dn > 0:
        it_tmp = it.copy()
        it_tmp[it_tmp < dn] = 0
        it = it_tmp

    it = normalize(it, mode)
    flat(v, mz, it, precision, low, use_max)

    if de == 1:
        remove_precursor(v, mass_val, c, precision, low, r=1)

    max_v = 0.0
    if v.size > 0:
        max_v = np.max(v)
    if max_v > 0.0:
        v /= max_v
    return v

def vectorlize(mz, it, mass_val, c, precision, dim, low, mode, out=None, kth=-1, th=-1, de=1, dn=-1, use_max=0):
    if out is None:
        out = np.zeros(dim, dtype='float32')
    return native_vectorlize(asnp32(mz), np32(it), mass_val, c, precision, dim, low, mode, out, kth, th, de, dn, use_max)

# This section defines amino-acid dictionaries, token maps, and mass tables.
mono = {"G": 57.021464, "A": 71.037114, "S": 87.032029, "P": 97.052764, "V": 99.068414, "T": 101.04768,
        "C": 160.03019, "L": 113.08406, "I": 113.08406, "D": 115.02694, "Q": 128.05858, "K": 128.09496,
        "E": 129.04259, "M": 131.04048, "m": 147.0354, "H": 137.05891, "F": 147.06441, "R": 156.10111,
        "Y": 163.06333, "N": 114.04293, "W": 186.07931, "O": 147.03538, "Z": 147.0354}
mono = {k: v for k, v in sorted(mono.items(), key=lambda item: item[1])}

Alist = list('ACDEFGHIKLMNPQRSTVWYZ')
clist = ['*'] + Alist + [']', '[']
oh_dim = len(clist)

charMap = {aa: i for i, aa in enumerate(clist)}
idmap = {i: aa for i, aa in enumerate(clist)}
mlist = asnp32([0] + [mono[a] for a in Alist] + [0, 0])
PAD_TOKEN = '*'
END_TOKEN = ']'
START_TOKEN = '['
PAD_ID = charMap[PAD_TOKEN]
END_ID = charMap[END_TOKEN]
START_ID = charMap[START_TOKEN]
MAX_PEPTIDE_LENGTH = 30

# This section defines sequence-level features and one-hot encoders.
@nb.njit
def AA_pairs_native(seq, v):
    for i in range(len(seq) - 1):
        aa1 = seq[i]
        aa2 = seq[i + 1]
        if aa1 > 0 and aa2 > 0 and aa1 <= 20 and aa2 <= 20:
            v[(aa1 - 1) * 20 + aa2 - 1] = 1
    return v

def AA_pairs(seq, out=None):
    if out is None:
        out = np.zeros(400, dtype='float32')
    return AA_pairs_native(seq, out)

def compose(pep):
    cl = np.zeros(len(Alist), dtype='int32')
    for i, A in enumerate(Alist):
        cl[i] = pep.count(A)
    return cl

@nb.njit
def n_encode(seq, length, em):
    em[0][START_ID] = 1
    for i, aa in enumerate(seq):
        em[i + 1][aa] = 1
    em[len(seq) + 1][END_ID] = 1
    em[len(seq) + 2:, PAD_ID] = 1
    return em

def encode(seq, length=-1, out=None):
    if length <= 0:
        length = len(seq)
    out = np.zeros((length, oh_dim), dtype='float32') if out is None else out
    return n_encode(seq, length, out)

def toseq(pep):
    return np.int32([charMap[c] for c in pep.upper()])

# This section defines data loading, filtering, and MGF conversion logic.
def spectra_ok(sp, ppm_threshold=10):
    mz, mass_val, pep, c = sp['mz'], sp['mass'], sp['pep'], sp['charge']
    if not pep.isalpha():
        return False
    if len(pep) > MAX_PEPTIDE_LENGTH:
        return False
    try:
        if ppm_threshold > 0 and abs(ppmdiff(sp)) > ppm_threshold:
            return False
    except Exception:
        return False
    return True

def filter_spectra(db):
    return [sp for sp in db if spectra_ok(sp)]

cr = {1: 1, 2: 0.9, 3: 0.85, 4: 0.8, 5: 0.75, 6: 0.75, 7: 0.75, 8: 0.75}
types = {'un': 0, 'cid': 1, 'etd': 2, 'hcd': 3, 'ethcd': 4, 'etcid': 5}

def convert_mgf(sps):
    db = []
    for sp in sps:
        param = sp['params']
        if 'charge' not in param:
            raise ValueError("Missing charge in spectrum params")

        c = int(str(param['charge'][0])[0])
        title = str(param.get('title', f"spectrum_{len(db)}"))
        pep = str(param.get('seq', ''))

        if 'pepmass' in param:
            mass_val = param['pepmass'][0]
        else:
            mass_val = float(param['parent'])

        if 'hcd' in param:
            try:
                hcd = param['hcd']
                if hcd[-1] == '%':
                    hcd = float(hcd[:-1])
                elif hcd[-2:] == 'eV':
                    hcd = float(hcd[:-2])
                    hcd = hcd * 500 * cr[c] / mass_val
                else:
                    raise Exception("Invalid type")
            except:
                hcd = 0
        else:
            hcd = 0

        db.append({
            'pep': pep, 'charge': c, 'mass': mass_val, 'mz': sp['m/z array'], 'it': sp['intensity array'],
            'nmod': 0, 'mod': np.zeros(len(pep), 'int32'), 'nce': hcd, 'title': title
        })
    return db

def readmgf(fn, type):
    with open(fn, "r") as file:
        data = mgf.read(file, convert_arrays=1, read_charges=False, dtype='float32', use_index=False)
        codes = convert_mgf(data)
    for sp in codes:
        sp['type'] = types[type]
    return codes

def i2l(sps):
    sps = [sp.copy() for sp in sps]
    for sp in sps:
        sp['pep'] = sp['pep'].replace('I', 'L')
    return sps

# This section defines normalization/layer factories and residual blocks for backbone construction.
def norm_layer(norm):
    def get_norm_layer(**kws):
        if norm == 'bn':
            return BatchNormalization(**kws)
        elif norm == 'in':
            return GroupNormalization(groups=-1, **kws)
        elif norm == 'ln':
            return LayerNormalization(**kws)
        elif norm is None or norm == 'none':
            return lambda x: x
        raise ValueError(f"Unsupported norm type: {norm}")
    return get_norm_layer

def layerset(c2d=0, norm=None, ghost=0):
    if c2d:
        return config({
            'ConvLayer': k.layers.Conv2D,
            'UpSamplingLayer': k.layers.UpSampling2D,
            'MaxPoolingLayer': k.layers.MaxPooling2D,
            'AveragePoolingLayer': k.layers.AveragePooling2D,
            'GlobalPoolingLayer': k.layers.GlobalAveragePooling2D,
            'GlobalMaxLayer': k.layers.GlobalMaxPooling2D,
            'ZeroPadding': k.layers.ZeroPadding2D,
            'c2d': c2d,
            'normalizer': norm_layer(norm)
        })
    return config({
        'ConvLayer': k.layers.Conv1D,
        'UpSamplingLayer': k.layers.UpSampling1D,
        'MaxPoolingLayer': k.layers.MaxPooling1D,
        'AveragePoolingLayer': k.layers.AveragePooling1D,
        'GlobalPoolingLayer': k.layers.GlobalAveragePooling1D,
        'GlobalMaxLayer': k.layers.GlobalMaxPooling1D,
        'ZeroPadding': k.layers.ZeroPadding1D,
        'c2d': c2d,
        'normalizer': norm_layer(norm)
    })

def merge(o1, c1, c2d=0, strides=1, lset={}, norm=None, mact=None):
    lset = layerset(c2d, norm, **lset)
    layers = K.int_shape(c1)[-1]

    if strides > 1 or K.int_shape(o1)[-1] != layers:
        if strides > 1:
            o1 = lset.ZeroPadding((0, strides - 1))(o1)
            o1 = lset.AveragePoolingLayer(strides)(o1)
        if K.int_shape(o1)[-1] != layers:
            o1 = lset.ConvLayer(layers, kernel_size=1, padding='same')(o1)
        o1 = lset.normalizer()(o1)

    return Add()([o1, c1]) if mact is None else Activation(mact)(Add()([o1, c1]))

def conv(x, layers, kernel, c2d=0, act='elu', lset={}, norm=None, dilation_rate=1, tcn=1, strides=1, se=0, **kws):
    lset = layerset(c2d, norm, **lset)
    if isinstance(kernel, int):
        kernel = (kernel,)
    for i, ks in enumerate(kernel):
        if i > 0:
            x = Activation(act)(x)
        x = lset.ConvLayer(layers, kernel_size=ks, padding='same', strides=strides, dilation_rate=dilation_rate, **kws)(x)
        x = lset.normalizer()(x)
        for r in range(1, tcn):
            assert strides == 1 and dilation_rate == 1
            x = Activation(act)(x)
            x = lset.ConvLayer(layers, kernel_size=kernel, padding='same', dilation_rate=2 ** r, **kws)(x)
            x = lset.normalizer()(x)
    return x

def res(x, l, ks, add=1, act='relu', c2d=False, norm=None, pool=2, strides=1, pooling='nil', lset={}, **kws):
    if not c2d:
        assert len(x.shape) == 3
    else:
        assert len(x.shape) == 4

    if pooling == 'up':
        x = lset.UpSamplingLayer(pool)(x)

    xc = conv(x, l, ks, c2d=c2d, lset=lset, norm=norm, act=act, strides=strides, **kws)
    if add:
        xc = merge(x, xc, c2d=c2d, lset=lset, norm=norm, mact=None, strides=strides)
    x = Activation(act)(xc)

    if pooling == 1 or pooling == 'max':
        x = lset.MaxPoolingLayer(pool)(x)
    elif pooling == 2 or pooling == 'ave':
        x = lset.AveragePoolingLayer(pool)(x)
    return x

# This section sanitizes arrays to remove NaN/Inf before passing data into model training.
def sanitize_tuple_of_arrays(tup):
    if isinstance(tup, (list, tuple)):
        out = []
        for arr in tup:
            if isinstance(arr, np.ndarray):
                arr = np.nan_to_num(arr, nan=0.0, posinf=1e6, neginf=-1e6)
            out.append(arr)
        return tuple(out)
    elif isinstance(tup, np.ndarray):
        return np.nan_to_num(tup, nan=0.0, posinf=1e6, neginf=-1e6)
    else:
        return tup

# This section defines hyperparameters and input schema.
class hyper_para():
    @dataclass(frozen=True)
    class hyper():
        lmax: int = 30
        outlen: int = lmax + 2
        m1max: int = -1
        mz_max: int = 2048
        pre: float = 0.1
        low: float = 0
        dim: int = int(mz_max / pre)
        sp_dim: int = 4
        maxc: int = 8
        mode: int = 3
        kth: int = 50
        dynamic = config({'enhance': 1, 'bsz': 512})
        inputs = config({
            'y': ([sp_dim, dim], 'float32'),
            'info': ([2], 'float32'),
            'charge': ([maxc], 'float32')
        })

    def __init__(self):
        self.inner = self.__class__.hyper()

    def __getattr__(self, att):
        return getattr(self.inner, att)

    def dict(self):
        return asdict(self.inner)

# This section defines end-to-end preprocessing for model inputs and multitask labels.
class data_processor():
    def __init__(self, hyper):
        self.hyper = hyper

    def data_enhance(self, mzs, its):
        its = normalize(its, self.hyper.mode)
        scale = 0.95 + np.random.rand() * 0.1
        its = its * scale
        noise_level = np.random.rand() * 0.005
        its = its + np.random.randn(*its.shape) * noise_level * np.max(its)
        its = np.clip(its, 0, None)

        if len(mzs) > 80:
            th = kth(its, int(abs(clipn(sigma=2)) * self.hyper.kth))
            mask = its > th
            mzs, its = mzs[mask], its[mask]

        if len(its) > 20 and np.random.rand() < 0.05:
            idx = np.random.choice(len(its), size=max(1, int(len(its) * 0.01)), replace=False)
            its[idx] = 0
        return mzs, its

    def get_inputs(self, sps, training=1):
        hyper = self.hyper
        batch_size = len(sps)
        inputs = config({})
        for spec in hyper.inputs:
            inputs[spec] = np.zeros((batch_size, *hyper.inputs[spec][0]), dtype=hyper.inputs[spec][1])

        for i, sp in enumerate(sps):
            mass_val, c, mzs, its = sp['mass'], sp['charge'], sp['mz'], sp['it']
            mzs = mzs / 1.00052
            if training and hyper.dynamic.enhance:
                mzs, its = self.data_enhance(mzs, its)
            else:
                its = normalize(its, self.hyper.mode)

            inputs.info[i][0] = mass_val / hyper.mz_max
            inputs.info[i][1] = sp['type']
            inputs.charge[i][c - 1] = 1

            precursor_index = min(hyper.dim - 1, round((mass_val * c - c + 1) / hyper.pre))
            vectorlize(mzs, its, mass_val, c, hyper.pre, hyper.dim, hyper.low, 0, out=inputs.y[i][0], use_max=1)
            inputs.y[i][1][:precursor_index] = inputs.y[i][0][:precursor_index][::-1]
            vectorlize(mzs, its, mass_val, c, hyper.pre, hyper.dim, hyper.low, 0, out=inputs.y[i][2], use_max=0)
            inputs.y[i][3][:precursor_index] = inputs.y[i][2][:precursor_index][::-1]
        return tuple([inputs[key] for key in inputs])

    def process(self, sps, training=1):
        hyper = self.hyper
        batch_size = len(sps)

        rst = config({
            'peps': np.zeros((batch_size, hyper.outlen, oh_dim), dtype='float32'),
            'main_prob': np.zeros((batch_size, hyper.outlen, oh_dim), dtype='float32'),
            'diff_output': np.zeros((batch_size, hyper.outlen, oh_dim), dtype='float32')
        })
        mtl = config({
            'exist': ([len(Alist)], 'float32'),
            'nums': ([len(Alist)], 'float32'),
            'di': ([400], 'float32'),
            'length': ([hyper.outlen], 'float32'),
            'rk': ([1], 'float32'),
            'charge': ([1], 'int32'),
            'mass': ([1], 'float32')
        })

        for task in mtl:
            rst[task] = np.zeros((batch_size, *mtl[task][0]), dtype=mtl[task][1])

        for i, sp in enumerate(sps):
            pep, mass_val, c, mzs, its = sp['pep'], sp['mass'], sp['charge'], sp['mz'], sp['it']
            pep = pep.upper().replace('I', 'L')
            seq = toseq(pep)

            encode(seq, out=rst.peps[i])
            rst.main_prob[i] = rst.peps[i].copy()
            rst.diff_output[i] = rst.peps[i].copy()

            rst.mass[i] = mass_val / 4000
            rst.charge[i] = c - 1

            max_length = rst.length.shape[1]
            if len(pep) < max_length:
                rst.length[i][len(pep)] = 1
            else:
                rst.length[i][max_length - 1] = 1

            rst.rk[i] = int(pep[-1] == 'R' or pep[-1] == 'K')
            AA_pairs(seq, out=rst.di[i])
            rst.nums[i] = compose(pep)
            for aa in pep:
                rst.exist[i][charMap[aa] - 1] = 1

        inputs = sanitize_tuple_of_arrays(self.get_inputs(sps, training=training))
        for kk in rst:
            if isinstance(rst[kk], np.ndarray):
                rst[kk] = np.nan_to_num(rst[kk], nan=0.0, posinf=1e6, neginf=-1e6)

        return (inputs, {key: rst[key] for key in rst})

hyper = hyper_para()
processor = data_processor(hyper)

# This section defines feature pyramid merging and SE attention block modules.
def bottomup(fu, norm='in', act='relu', **kws):
    v1 = fu[0]
    fu = fu[1:]
    for u in fu:
        v1 = res(v1, K.int_shape(u)[-1], 5, act=act, strides=2, norm=norm, add=0, **kws)
        v1 = k.layers.Add()([v1, u])
    return v1

class SEBlock(Layer):
    def __init__(self, ratio=8, **kwargs):
        super(SEBlock, self).__init__(**kwargs)
        self.ratio = ratio

    def build(self, input_shape):
        filters = input_shape[-1]
        self.squeeze = GlobalAveragePooling1D()
        self.excitation = k.Sequential([
            Dense(filters // self.ratio, activation='relu', kernel_initializer='he_normal'),
            Dense(filters, activation='sigmoid', kernel_initializer='he_normal')
        ])
        super(SEBlock, self).build(input_shape)

    def call(self, inputs):
        x = self.squeeze(inputs)
        x = self.excitation(x)
        x = tf.expand_dims(x, axis=1)
        return inputs * x

# This section defines the denovo architecture, including auxiliary heads and diffusion fusion output.
class denovo_model():
    @staticmethod
    def sp_net(hyper, act='relu', norm='in'):
        inp = Input(shape=hyper.inputs['y'][0], name='sub_sp_inp')
        mz_inp = Input(shape=hyper.inputs['info'][0], name='sub_mz_inp')
        c_inp = Input(shape=hyper.inputs['charge'][0], name='sub_charge_inp')
        v1 = k.layers.Permute((2, 1))(inp)

        def sub_net(v1, act='relu'):
            def se_residual_block(inputs, channels, kernel_size, dilation_depth):
                features = conv(
                    inputs,
                    channels,
                    kernel_size,
                    tcn=dilation_depth,
                    norm=norm,
                    act=act,
                )
                features = SEBlock()(features)
                outputs = merge(inputs, features, norm=norm)
                return Activation(act)(outputs)

            for i, l in enumerate([8, 12]):
                v1 = res(v1, l, 7, norm=norm, add=1, act=act, strides=2)

            lst = []
            fils = np.int32([16, 24, 32, 48, 64]) * 12
            tcn = np.int32([8, 7, 6, 5, 4]) + 1

            for i, (l, r) in enumerate(zip(fils, tcn)):
                if i > 0:
                    v1 = res(v1, l, 9, norm=norm, add=1, act=act, strides=2)

                ext = r - 5
                if r > ext:
                    r = r - ext
                    ks = int(5 * 2 ** ext) - 1
                else:
                    ks = int(5 * 2 ** int(r - 1)) - 1
                    r = 1

                v1 = se_residual_block(v1, l, ks, r)
                lst.append(v1)
            return v1, lst

        v1, lst = sub_net(v1)
        v1 = bottomup(lst[2:])
        v1 = k.layers.Permute((2, 1))(v1)
        v1 = res(v1, hyper.outlen, 1, act=act, norm=norm, add=0)
        v1 = k.layers.Permute((2, 1))(v1)

        l_size = K.int_shape(v1)[-2]
        infos = k.layers.Concatenate(axis=-1)([mz_inp, c_inp])
        infos = k.layers.Reshape((l_size, 1))(k.layers.Dense(l_size, activation='sigmoid')(infos))
        v1 = k.layers.Concatenate(axis=-1)([v1, infos])

        return k.models.Model([inp, mz_inp, c_inp], v1, name='sp_net')

    @staticmethod
    def auxiliary_tasks(v1, hyper, act='relu', norm='in'):
        x = k.layers.GlobalAveragePooling1D(name='exist_gap')(v1)
        x = Dropout(0.3)(x)
        exist = Dense(len(Alist), name='exist_fc', kernel_regularizer=regularizers.l2(1e-5))(x)
        exist_out = Activation('sigmoid', name='exist')(exist)

        length_conv = Conv1D(hyper.outlen, 1, name='length_conv')(v1)
        length_gap = GlobalAveragePooling1D(name='length_gap')(length_conv)
        length_out = Activation('softmax', name='length')(length_gap)

        charge_fc = Dense(hyper.maxc, name='charge_fc')(x)
        charge_out = Activation('softmax', name='charge')(charge_fc)

        rk_out = Dense(1, activation='sigmoid', name='rk')(x)

        di_fc = Dense(400, name='di_fc')(x)
        di_out = Activation('sigmoid', name='di')(di_fc)

        mass_out = Dense(1, name='mass')(x)

        x = k.layers.Permute((2, 1))(v1)
        x = res(x, len(Alist), 1, name='nums_res')
        x = k.layers.Permute((2, 1))(x)
        nums_conv = Conv1D(hyper.lmax, 1, name='nums_conv')(x)
        nums_out = Activation('softmax', name='nums')(nums_conv)

        return [exist_out, length_out, charge_out, rk_out, di_out, mass_out, nums_out]

    @staticmethod
    def build(hyper, act='relu', norm='in'):
        inp = Input(shape=hyper.inputs['y'][0], name='sp_inp')
        mz_inp = Input(shape=hyper.inputs['info'][0], name='mz_inp')
        c_inp = Input(shape=hyper.inputs['charge'][0], name='charge_inp')
        model_inputs = [inp, mz_inp, c_inp]

        spmodel = denovo_model.sp_net(hyper)
        sp_vector = spmodel(model_inputs)
        sp_vector = SpatialDropout1D(0.3)(sp_vector)

        aux_outputs = denovo_model.auxiliary_tasks(sp_vector, hyper)

        main_output = Conv1D(oh_dim, 1, padding='same', name='main_conv',
                             kernel_regularizer=regularizers.l2(1e-4))(sp_vector)
        main_prob = Activation('softmax', name='main_prob')(main_output)

        diff_model = DiffusionModel(seq_length=hyper.outlen, channels=oh_dim)
        diff_output = diff_model(main_prob)
        diff_output = Activation('softmax', name='diff_output')(diff_output)

        fused = Add(name='fusion')([main_prob, diff_output])
        final_pep = Activation('softmax', name='peps')(fused)

        full_model = Model(
            inputs=model_inputs,
            outputs=aux_outputs + [final_pep, main_prob, diff_output],
            name='full_model_with_diffusion'
        )
        novo = Model(inputs=model_inputs, outputs=final_pep, name='denovo')
        return full_model, novo, spmodel

# This section defines custom losses, metrics, and compilation logic.
class model_builder():
    class loss_fn:
        @staticmethod
        def mse_ce(fn=k.losses.categorical_crossentropy, c=0.25):
            def mse(yt, yp):
                yt = K.cast(yt, 'float32')
                yp = K.cast(yp, 'float32')
                yp = K.clip(yp, K.epsilon(), 1.0 - K.epsilon())
                ce = fn(yt, yp)
                yt_idx = K.cast(K.argmax(yt, axis=-1), 'float32')
                yp_idx = K.cast(K.argmax(yp, axis=-1), 'float32')
                return c * k.losses.mean_absolute_error(yt_idx / 32.0, yp_idx / 32.0) + ce * 0.25
            return mse

        @staticmethod
        def mask_ce_with_mass(yt, yp, ls=0.1, mass_weight=0.01):
            yt = K.cast(yt, 'float32')
            yp = K.cast(yp, 'float32')
            yp = K.clip(yp, K.epsilon(), 1.0 - K.epsilon())

            yts = K.argmax(yt, axis=-1)
            mask = K.cast(K.greater(yts, 0), dtype='float32')

            ce_loss_val = k.losses.categorical_crossentropy(yt, yp, label_smoothing=ls)
            ce_loss_val = ce_loss_val * mask
            ce_loss = K.sum(ce_loss_val, axis=-1) / (K.sum(mask, axis=-1) + K.epsilon())

            mlist_t = tf.convert_to_tensor(mlist, dtype=tf.float32)
            pred_mass = K.sum(yp * mlist_t, axis=-1)
            true_mass = K.sum(yt * mlist_t, axis=-1)

            mass_diff = K.abs(pred_mass - true_mass) * mask
            mass_loss = K.sum(mass_diff, axis=-1) / (K.sum(mask, axis=-1) + K.epsilon())
            return ce_loss + mass_weight * mass_loss

        @staticmethod
        def mass_ce(fn=k.losses.categorical_crossentropy, c=0.001):
            def mse(yt, yp):
                yt = K.cast(yt, 'float32')
                yp = K.cast(yp, 'float32')
                yp = K.clip(yp, K.epsilon(), 1.0 - K.epsilon())
                ce_loss = fn(yt, yp)
                mlist_t = K.cast(mlist, 'float32')
                mp = K.sum(K.batch_flatten(yp * mlist_t), axis=-1)
                mt = K.sum(K.batch_flatten(yt * mlist_t), axis=-1)
                return c * k.losses.mean_absolute_error(mp, mt) + ce_loss
            return mse

        @staticmethod
        def mask_ce(yt, yp, ls=0.00):
            yt = K.cast(yt, 'float32')
            yp = K.cast(yp, 'float32')
            yp = K.clip(yp, K.epsilon(), 1.0 - K.epsilon())
            yts = K.argmax(yt, axis=-1)
            mask = K.cast(K.greater(yts, 0), dtype='float32')
            loss = k.losses.categorical_crossentropy(yt, yp, label_smoothing=ls) * mask
            return K.sum(loss, axis=-1) / (K.sum(mask, axis=-1) + K.epsilon())

        @staticmethod
        def mask_acc(yt, yp):
            yts = K.argmax(yt, axis=-1)
            yps = K.argmax(yp, axis=-1)
            mask = K.cast(K.greater(yts, 0), dtype='float32')
            err = K.cast(K.not_equal(yts, yps), dtype='float32') * mask
            return 1.0 - K.sum(err, axis=-1) / (K.sum(mask, axis=-1) + K.epsilon())

        @staticmethod
        def full_acc(yt, yp):
            yts = K.argmax(yt, axis=-1)
            yps = K.argmax(yp, axis=-1)
            mask = K.cast(K.greater(yts, 0), dtype='bool')
            equal_elements = tf.equal(yts, yps)
            is_valid_pos = tf.logical_or(equal_elements, tf.logical_not(mask))
            return K.cast(K.all(is_valid_pos, axis=-1), dtype='float32')

    def __init__(self, hyper, denovo_model=denovo_model):
        self.hyper = hyper
        self.param = config()
        self.loss_fn = self.__class__.loss_fn
        self.denovo_model = denovo_model
        self.hyper.aa_mass_list = tf.constant(mlist, dtype=tf.float32)
        self.loss_weights = None

    def set_loss(self):
        self.losses = {
            "exist": binary_crossentropy,
            "length": self.loss_fn.mse_ce(categorical_crossentropy),
            "charge": sparse_categorical_crossentropy,
            "rk": binary_crossentropy,
            "di": binary_crossentropy,
            "mass": MSE,
            "nums": sparse_categorical_crossentropy,
            "peps": self.loss_fn.mask_ce_with_mass,
            "main_prob": self.loss_fn.mask_ce,
            "diff_output": self.loss_fn.mask_ce
        }

        self.weights = {
            "exist": 0.1, "length": 0.05, "charge": 0.05, "rk": 0.05, "di": 0.001,
            "mass": 0.1, "nums": 0.1, "peps": 2.0, "main_prob": 1.0, "diff_output": 1.0
        }

        def masked_categorical_accuracy(y_true, y_pred):
            y_true_ids = K.argmax(K.cast(y_true, 'float32'), axis=-1)
            y_pred_ids = K.argmax(K.cast(y_pred, 'float32'), axis=-1)
            mask = K.cast(K.greater(y_true_ids, 0), K.floatx())
            matches = K.cast(K.equal(y_true_ids, y_pred_ids), K.floatx())
            return K.sum(matches * mask) / (K.sum(mask) + K.epsilon())

        self.metrics = {
            "peps": [masked_categorical_accuracy, self.loss_fn.mask_ce, self.loss_fn.mask_acc, self.loss_fn.full_acc],
            "main_prob": [self.loss_fn.mask_acc],
            "diff_output": [self.loss_fn.mask_acc],
            "nums": 'acc',
            "exist": 'binary_accuracy',
            "rk": 'acc',
            "di": 'binary_accuracy'
        }

    def build(self, summary=True):
        self.dm, self.novo, *self.rest = self.denovo_model.build(self.hyper)
        if summary:
            self.rest[0].summary()
        return self.dm, self.novo

    def compile(self, model=None, opt='adam'):
        self.set_loss()

        output_names = ['exist', 'length', 'charge', 'rk', 'di', 'mass', 'nums', 'peps', 'main_prob', 'diff_output']
        loss_weights = [self.weights[name] for name in output_names]
        losses = [self.losses[name] for name in output_names]
        metrics = [self.metrics.get(name, []) for name in output_names]
        self.loss_weights = loss_weights

        target = self.dm if model is None else model
        print(f"Compiling model with {len(target.outputs)} outputs and {len(losses)} losses.")
        target.compile(optimizer=opt, loss=losses, loss_weights=loss_weights, metrics=metrics)

# This section sets deterministic seeds for reproducible experiments.
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

# This section defines the training manager for standard train-set + test-set workflow.
class train_mgr():
    def data_generator(self, sps, **kws):
        return data_seq(sps, processor.process, hyper.dynamic.bsz, xonly=0, **kws)

    def setup(self, output_dir='output', **kws):
        self.output_dir = os.path.abspath(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)
        log_save_path = os.path.join(self.output_dir, 'training_log.csv')
        checkpoint_path = os.path.join(self.output_dir, 'denovo_best.weights.h5')
        self.his = k.callbacks.History()
        self.callbacks = [
            ModelCheckpoint(checkpoint_path, save_best_only=True, save_weights_only=True,
                             monitor='val_peps_mask_acc', mode='max', verbose=1),
            ReduceLROnPlateau(monitor='val_peps_mask_acc', factor=0.5, patience=5, min_lr=1e-6, verbose=1, mode='max'),
            EarlyStopping(monitor='val_peps_mask_acc', mode='max', patience=6, min_delta=1e-4, restore_best_weights=True, verbose=1),
            CSVLogger(log_save_path, separator=',', append=True)
        ]
        self.builder = model_builder(hyper)
        self.dm, self.novo = self.builder.build(**kws)
        return self.dm, self.novo

    def prepare_data(self, train_mgf, val_mgf, frag_type='hcd'):
        print(f"[Data] Standard train/validation mode. Train={train_mgf}, Validation={val_mgf}, Frag={frag_type}")
        self.trainingset = i2l(filter_spectra(readmgf(train_mgf, frag_type)))
        self.valset = i2l(filter_spectra(readmgf(val_mgf, frag_type)))
        if not self.trainingset:
            raise ValueError("No valid training spectra remained after preprocessing.")
        if not self.valset:
            raise ValueError("No valid validation spectra remained after preprocessing.")
        print(f"[Data] Train spectra={len(self.trainingset)}, Validation spectra={len(self.valset)}")

    def compile(self, bsz=64, lr=None, epochs=50):
        hyper.dynamic.eps = int(epochs)
        hyper.dynamic.bsz = 64 if bsz is None else bsz
        hyper.dynamic.lr = lr if lr else (hyper.dynamic.bsz / 1024) * 0.0009 * 16 * 6
        hyper.dynamic.opt = Adam(learning_rate=hyper.dynamic.lr)

        print("=" * 60)
        print("Model outputs:")
        for i, out in enumerate(self.dm.outputs):
            print(f"  {i}: {out.name}")
        print("=" * 60)

        self.builder.compile(model=self.dm, opt=hyper.dynamic.opt)

    def run(self):
        print("\nCreating data generators...")
        train_gen = self.data_generator(self.trainingset, shuffle=1, training=1)
        val_gen = self.data_generator(self.valset, shuffle=0, training=0)

        print(f"Train batches: {len(train_gen)}")
        print(f"Test batches: {len(val_gen)}")

        try:
            test_x, test_y = train_gen[0]
            print(f"Input shapes: {[x.shape for x in test_x]}")
            print(f"Label keys: {list(test_y.keys())}")
        except Exception as e:
            print(f"Data generator error: {e}")
            return

        print("\nStarting training...")
        self.dm.fit(
            train_gen,
            validation_data=val_gen,
            epochs=hyper.dynamic.eps,
            callbacks=self.callbacks + [self.his],
            verbose=1
        )

        gc.collect()
        print("Training finished.")

# This section provides CLI entrypoint for RTX 5090 standard train+test training.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DenNovo from separate training and validation MGF files.')
    parser.add_argument('--train-mgf', '--train_mgf', dest='train_mgf', required=True, help='Training MGF path')
    parser.add_argument('--val-mgf', '--val_mgf', '--test_mgf', dest='val_mgf', required=True, help='Validation MGF path')
    parser.add_argument('--frag', type=str, default='hcd', choices=list(types.keys()), help='Fragmentation type')
    parser.add_argument('--batch-size', '--bsz', dest='batch_size', type=int, default=64, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Maximum number of epochs')
    parser.add_argument('--learning-rate', '--lr', dest='learning_rate', type=float, default=None, help='Optional initial learning rate')
    parser.add_argument('--output-dir', type=str, default='output', help='Directory for checkpoints and logs')
    args = parser.parse_args()

    manager = train_mgr()
    dm, novo = manager.setup(summary=1, output_dir=args.output_dir)
    manager.compile(bsz=args.batch_size, lr=args.learning_rate, epochs=args.epochs)
    manager.prepare_data(train_mgf=args.train_mgf, val_mgf=args.val_mgf, frag_type=args.frag)
    manager.run()
