#################################
# preprocess registered data: butterworth -> standardization -> adc
# input: registered raw data
# output: registered/preprocessed data
#################################

import numpy as np
import scipy
from tqdm import tqdm
import os

from yass.preprocess.util import _butterworth, get_std, _standardize

from yass.preprocess.util import merge_filtered_files

def preprocess(reader, sampling_rate=30000, 
               apply_filter=True, 
               low_frequency=300, 
               high_factor=0.1, 
               order=3,
               neuropixel_version=2,
               run_adc_shift=True):

    chunk_5sec = 5*sampling_rate
    small_batch = reader.read_data(
        data_start=reader.rec_len//2 - chunk_5sec//2,
        data_end=reader.rec_len//2 + chunk_5sec//2)
    
    mean_sd_directory = os.path.join('.', "mean_sd")
    if not os.path.exists(mean_sd_directory):
        os.makedirs(mean_sd_directory)
    
    fname_mean_sd = os.path.join(
        mean_sd_directory, 'mean_and_standard_dev_value.npz')
    if not os.path.exists(fname_mean_sd):
        get_std(small_batch, sampling_rate,
                fname_mean_sd, apply_filter=apply_filter, low_frequency=low_frequency,
                high_factor=high_factor,
                order=order)
        
    preprocessed_location = os.path.join('.', "preprocessed_files")
    if not os.path.exists(preprocessed_location):
        os.makedirs(preprocessed_location)
        
    mean_sd = np.load(fname_mean_sd)
    sd = mean_sd['sd']
    centers = mean_sd['centers']

    h = trace_header(version=neuropixel_version)

    for batch_id in tqdm(range(reader.n_batches)):
        ts = reader.read_data_batch(batch_id, add_buffer=False)
        
        # butterworth
        if apply_filter:
            ts = _butterworth(ts, low_frequency, high_factor, order, sampling_rate)
        
        # standardization
        ts = _standardize(ts, sd, centers)

        if run_adc_shift:
            # adc shift
            ts = fshift(ts.T, h['sample_shift'], axis=1)
            ts = ts.T
        
        # save
        fname = os.path.join(
            preprocessed_location,
            "preprocessed_{}.npy".format(
                str(batch_id).zfill(6)))
        np.save(fname, ts.astype(np.float32))
        
    # merge preprocessed files
    merge_filtered_files(preprocessed_location, preprocessed_location, delete = True)

    
#########################################################################
# citation: https://github.com/int-brain-lab/ibllib/tree/sigproc
#########################################################################
def fcn_cosine(bounds):
    """
    Returns a soft thresholding function with a cosine taper:
    values <= bounds[0]: values
    values < bounds[0] < bounds[1] : cosine taper
    values < bounds[1]: bounds[1]
    :param bounds:
    :return: lambda function
    """
    def _cos(x):
        return (1 - np.cos((x - bounds[0]) / (bounds[1] - bounds[0]) * np.pi)) / 2
    func = lambda x: _fcn_extrap(x, _cos, bounds)  # noqa
    return func

TIP_SIZE_UM = 200

SYNC_PIN_OUT = {'3A': {"pin01": 0,
                       "pin02": 1,
                       "pin03": 2,
                       "pin04": 3,
                       "pin05": None,
                       "pin06": 4,
                       "pin07": 5,
                       "pin08": 6,
                       "pin09": 7,
                       "pin10": None,
                       "pin11": 8,
                       "pin12": 9,
                       "pin13": 10,
                       "pin14": 11,
                       "pin15": None,
                       "pin16": 12,
                       "pin17": 13,
                       "pin18": 14,
                       "pin19": 15,
                       "pin20": None,
                       "pin21": None,
                       "pin22": None,
                       "pin23": None,
                       "pin24": None
                       },
                '3B': {"P0.0": 0,
                       "P0.1": 1,
                       "P0.2": 2,
                       "P0.3": 3,
                       "P0.4": 4,
                       "P0.5": 5,
                       "P0.6": 6,
                       "P0.7": 7,
                       }
                }

# after moving to ks2.5, this should be deprecated
SITES_COORDINATES = np.array([
    [43., 20.],
    [11., 20.],
    [59., 40.],
    [27., 40.],
    [43., 60.],
    [11., 60.],
    [59., 80.],
    [27., 80.],
    [43., 100.],
    [11., 100.],
    [59., 120.],
    [27., 120.],
    [43., 140.],
    [11., 140.],
    [59., 160.],
    [27., 160.],
    [43., 180.],
    [11., 180.],
    [59., 200.],
    [27., 200.],
    [43., 220.],
    [11., 220.],
    [59., 240.],
    [27., 240.],
    [43., 260.],
    [11., 260.],
    [59., 280.],
    [27., 280.],
    [43., 300.],
    [11., 300.],
    [59., 320.],
    [27., 320.],
    [43., 340.],
    [11., 340.],
    [59., 360.],
    [27., 360.],
    [11., 380.],
    [59., 400.],
    [27., 400.],
    [43., 420.],
    [11., 420.],
    [59., 440.],
    [27., 440.],
    [43., 460.],
    [11., 460.],
    [59., 480.],
    [27., 480.],
    [43., 500.],
    [11., 500.],
    [59., 520.],
    [27., 520.],
    [43., 540.],
    [11., 540.],
    [59., 560.],
    [27., 560.],
    [43., 580.],
    [11., 580.],
    [59., 600.],
    [27., 600.],
    [43., 620.],
    [11., 620.],
    [59., 640.],
    [27., 640.],
    [43., 660.],
    [11., 660.],
    [59., 680.],
    [27., 680.],
    [43., 700.],
    [11., 700.],
    [59., 720.],
    [27., 720.],
    [43., 740.],
    [11., 740.],
    [59., 760.],
    [43., 780.],
    [11., 780.],
    [59., 800.],
    [27., 800.],
    [43., 820.],
    [11., 820.],
    [59., 840.],
    [27., 840.],
    [43., 860.],
    [11., 860.],
    [59., 880.],
    [27., 880.],
    [43., 900.],
    [11., 900.],
    [59., 920.],
    [27., 920.],
    [43., 940.],
    [11., 940.],
    [59., 960.],
    [27., 960.],
    [43., 980.],
    [11., 980.],
    [59., 1000.],
    [27., 1000.],
    [43., 1020.],
    [11., 1020.],
    [59., 1040.],
    [27., 1040.],
    [43., 1060.],
    [11., 1060.],
    [59., 1080.],
    [27., 1080.],
    [43., 1100.],
    [11., 1100.],
    [59., 1120.],
    [27., 1120.],
    [11., 1140.],
    [59., 1160.],
    [27., 1160.],
    [43., 1180.],
    [11., 1180.],
    [59., 1200.],
    [27., 1200.],
    [43., 1220.],
    [11., 1220.],
    [59., 1240.],
    [27., 1240.],
    [43., 1260.],
    [11., 1260.],
    [59., 1280.],
    [27., 1280.],
    [43., 1300.],
    [11., 1300.],
    [59., 1320.],
    [27., 1320.],
    [43., 1340.],
    [11., 1340.],
    [59., 1360.],
    [27., 1360.],
    [43., 1380.],
    [11., 1380.],
    [59., 1400.],
    [27., 1400.],
    [43., 1420.],
    [11., 1420.],
    [59., 1440.],
    [27., 1440.],
    [43., 1460.],
    [11., 1460.],
    [59., 1480.],
    [27., 1480.],
    [43., 1500.],
    [11., 1500.],
    [59., 1520.],
    [43., 1540.],
    [11., 1540.],
    [59., 1560.],
    [27., 1560.],
    [43., 1580.],
    [11., 1580.],
    [59., 1600.],
    [27., 1600.],
    [43., 1620.],
    [11., 1620.],
    [59., 1640.],
    [27., 1640.],
    [43., 1660.],
    [11., 1660.],
    [59., 1680.],
    [27., 1680.],
    [43., 1700.],
    [11., 1700.],
    [59., 1720.],
    [27., 1720.],
    [43., 1740.],
    [11., 1740.],
    [59., 1760.],
    [27., 1760.],
    [43., 1780.],
    [11., 1780.],
    [59., 1800.],
    [27., 1800.],
    [43., 1820.],
    [11., 1820.],
    [59., 1840.],
    [27., 1840.],
    [43., 1860.],
    [11., 1860.],
    [59., 1880.],
    [27., 1880.],
    [11., 1900.],
    [59., 1920.],
    [27., 1920.],
    [43., 1940.],
    [11., 1940.],
    [59., 1960.],
    [27., 1960.],
    [43., 1980.],
    [11., 1980.],
    [59., 2000.],
    [27., 2000.],
    [43., 2020.],
    [11., 2020.],
    [59., 2040.],
    [27., 2040.],
    [43., 2060.],
    [11., 2060.],
    [59., 2080.],
    [27., 2080.],
    [43., 2100.],
    [11., 2100.],
    [59., 2120.],
    [27., 2120.],
    [43., 2140.],
    [11., 2140.],
    [59., 2160.],
    [27., 2160.],
    [43., 2180.],
    [11., 2180.],
    [59., 2200.],
    [27., 2200.],
    [43., 2220.],
    [11., 2220.],
    [59., 2240.],
    [27., 2240.],
    [43., 2260.],
    [11., 2260.],
    [59., 2280.],
    [43., 2300.],
    [11., 2300.],
    [59., 2320.],
    [27., 2320.],
    [43., 2340.],
    [11., 2340.],
    [59., 2360.],
    [27., 2360.],
    [43., 2380.],
    [11., 2380.],
    [59., 2400.],
    [27., 2400.],
    [43., 2420.],
    [11., 2420.],
    [59., 2440.],
    [27., 2440.],
    [43., 2460.],
    [11., 2460.],
    [59., 2480.],
    [27., 2480.],
    [43., 2500.],
    [11., 2500.],
    [59., 2520.],
    [27., 2520.],
    [43., 2540.],
    [11., 2540.],
    [59., 2560.],
    [27., 2560.],
    [43., 2580.],
    [11., 2580.],
    [59., 2600.],
    [27., 2600.],
    [43., 2620.],
    [11., 2620.],
    [59., 2640.],
    [27., 2640.],
    [11., 2660.],
    [59., 2680.],
    [27., 2680.],
    [43., 2700.],
    [11., 2700.],
    [59., 2720.],
    [27., 2720.],
    [43., 2740.],
    [11., 2740.],
    [59., 2760.],
    [27., 2760.],
    [43., 2780.],
    [11., 2780.],
    [59., 2800.],
    [27., 2800.],
    [43., 2820.],
    [11., 2820.],
    [59., 2840.],
    [27., 2840.],
    [43., 2860.],
    [11., 2860.],
    [59., 2880.],
    [27., 2880.],
    [43., 2900.],
    [11., 2900.],
    [59., 2920.],
    [27., 2920.],
    [43., 2940.],
    [11., 2940.],
    [59., 2960.],
    [27., 2960.],
    [43., 2980.],
    [11., 2980.],
    [59., 3000.],
    [27., 3000.],
    [43., 3020.],
    [11., 3020.],
    [59., 3040.],
    [43., 3060.],
    [11., 3060.],
    [59., 3080.],
    [27., 3080.],
    [43., 3100.],
    [11., 3100.],
    [59., 3120.],
    [27., 3120.],
    [43., 3140.],
    [11., 3140.],
    [59., 3160.],
    [27., 3160.],
    [43., 3180.],
    [11., 3180.],
    [59., 3200.],
    [27., 3200.],
    [43., 3220.],
    [11., 3220.],
    [59., 3240.],
    [27., 3240.],
    [43., 3260.],
    [11., 3260.],
    [59., 3280.],
    [27., 3280.],
    [43., 3300.],
    [11., 3300.],
    [59., 3320.],
    [27., 3320.],
    [43., 3340.],
    [11., 3340.],
    [59., 3360.],
    [27., 3360.],
    [43., 3380.],
    [11., 3380.],
    [59., 3400.],
    [27., 3400.],
    [11., 3420.],
    [59., 3440.],
    [27., 3440.],
    [43., 3460.],
    [11., 3460.],
    [59., 3480.],
    [27., 3480.],
    [43., 3500.],
    [11., 3500.],
    [59., 3520.],
    [27., 3520.],
    [43., 3540.],
    [11., 3540.],
    [59., 3560.],
    [27., 3560.],
    [43., 3580.],
    [11., 3580.],
    [59., 3600.],
    [27., 3600.],
    [43., 3620.],
    [11., 3620.],
    [59., 3640.],
    [27., 3640.],
    [43., 3660.],
    [11., 3660.],
    [59., 3680.],
    [27., 3680.],
    [43., 3700.],
    [11., 3700.],
    [59., 3720.],
    [27., 3720.],
    [43., 3740.],
    [11., 3740.],
    [59., 3760.],
    [27., 3760.],
    [43., 3780.],
    [11., 3780.],
    [59., 3800.],
    [43., 3820.],
    [11., 3820.],
    [59., 3840.],
    [27., 3840.]])

NC = 384

def convolve(x, w, mode='full'):
    """
    Frequency domain convolution along the last dimension (2d arrays)
    Will broadcast if a matrix is convolved with a vector
    :param x:
    :param w:
    :return: convolution
    """
    nsx = x.shape[-1]
    nsw = w.shape[-1]
    ns = ns_optim_fft(nsx + nsw)
    x_ = np.concatenate((x, np.zeros([*x.shape[:-1], ns - nsx], dtype=x.dtype)), axis=-1)
    w_ = np.concatenate((w, np.zeros([*w.shape[:-1], ns - nsw], dtype=w.dtype)), axis=-1)
    xw = np.fft.irfft(np.fft.rfft(x_, axis=-1) * np.fft.rfft(w_, axis=-1), axis=-1)
    xw = xw[..., :(nsx + nsw)]  # remove 0 padding
    if mode == 'full':
        return xw
    elif mode == 'same':
        first = int(np.floor(nsw / 2)) - ((nsw + 1) % 2)
        last = int(np.ceil(nsw / 2)) + ((nsw + 1) % 2)
        return xw[..., first:-last]


def ns_optim_fft(ns):
    """
    Gets the next higher combination of factors of 2 and 3 than ns to compute efficient ffts
    :param ns:
    :return: nsoptim
    """
    p2, p3 = np.meshgrid(2 ** np.arange(25), 3 ** np.arange(15))
    sz = np.unique((p2 * p3).flatten())
    return sz[np.searchsorted(sz, ns)]


def dephas(w, phase, axis=-1):
    """
    dephas a signal by a given angle in degrees
    :param w:
    :param phase: phase in degrees
    :param axis:
    :return:
    """
    ns = w.shape[axis]
    W = freduce(np.fft.fft(w, axis=axis), axis=axis) * np.exp(- 1j * phase / 180 * np.pi)
    return np.real(np.fft.ifft(fexpand(W, ns=ns, axis=axis), axis=axis))


def fscale(ns, si=1, one_sided=False):
    """
    numpy.fft.fftfreq returns Nyquist as a negative frequency so we propose this instead
    :param ns: number of samples
    :param si: sampling interval in seconds
    :param one_sided: if True, returns only positive frequencies
    :return: fscale: numpy vector containing frequencies in Hertz
    """
    fsc = np.arange(0, np.floor(ns / 2) + 1) / ns / si  # sample the frequency scale
    if one_sided:
        return fsc
    else:
        return np.concatenate((fsc, -fsc[slice(-2 + (ns % 2), 0, -1)]), axis=0)


def freduce(x, axis=None):
    """
    Reduces a spectrum to positive frequencies only
    Works on the last dimension (contiguous in c-stored array)
    :param x: numpy.ndarray
    :param axis: axis along which to perform reduction (last axis by default)
    :return: numpy.ndarray
    """
    if axis is None:
        axis = x.ndim - 1
    siz = list(x.shape)
    siz[axis] = int(np.floor(siz[axis] / 2 + 1))
    return np.take(x, np.arange(0, siz[axis]), axis=axis)


def fexpand(x, ns=1, axis=None):
    """
    Reconstructs full spectrum from positive frequencies
    Works on the last dimension (contiguous in c-stored array)
    :param x: numpy.ndarray
    :param axis: axis along which to perform reduction (last axis by default)
    :return: numpy.ndarray
    """
    if axis is None:
        axis = x.ndim - 1
    # dec = int(ns % 2) * 2 - 1
    # xcomp = np.conj(np.flip(x[..., 1:x.shape[-1] + dec], axis=axis))
    ilast = int((ns + (ns % 2)) / 2)
    xcomp = np.conj(np.flip(np.take(x, np.arange(1, ilast), axis=axis), axis=axis))
    return np.concatenate((x, xcomp), axis=axis)


def bp(ts, si, b, axis=None):
    """
    Band-pass filter in frequency domain
    :param ts: time serie
    :param si: sampling interval in seconds
    :param b: cutout frequencies: 4 elements vector or list
    :param axis: axis along which to perform reduction (last axis by default)
    :return: filtered time serie
    """
    return _freq_filter(ts, si, b, axis=axis, typ='bp')


def lp(ts, si, b, axis=None):
    """
    Low-pass filter in frequency domain
    :param ts: time serie
    :param si: sampling interval in seconds
    :param b: cutout frequencies: 2 elements vector or list
    :param axis: axis along which to perform reduction (last axis by default)
    :return: filtered time serie
    """
    return _freq_filter(ts, si, b, axis=axis, typ='lp')


def hp(ts, si, b, axis=None):
    """
    High-pass filter in frequency domain
    :param ts: time serie
    :param si: sampling interval in seconds
    :param b: cutout frequencies: 2 elements vector or list
    :param axis: axis along which to perform reduction (last axis by default)
    :return: filtered time serie
    """
    return _freq_filter(ts, si, b, axis=axis, typ='hp')


def _freq_filter(ts, si, b, axis=None, typ='lp'):
    """
        Wrapper for hp/lp/bp filters
    """
    if axis is None:
        axis = ts.ndim - 1
    ns = ts.shape[axis]
    f = fscale(ns, si=si, one_sided=True)
    if typ == 'bp':
        filc = _freq_vector(f, b[0:2], typ='hp') * _freq_vector(f, b[2:4], typ='lp')
    else:
        filc = _freq_vector(f, b, typ=typ)
    if axis < (ts.ndim - 1):
        filc = filc[:, np.newaxis]
    return np.real(np.fft.ifft(np.fft.fft(ts, axis=axis) * fexpand(filc, ns, axis=0), axis=axis))


def _freq_vector(f, b, typ='lp'):
    """
        Returns a frequency modulated vector for filtering
        :param f: frequency vector, uniform and monotonic
        :param b: 2 bounds array
        :return: amplitude modulated frequency vector
    """
    filc = fcn_cosine(b)(f)
    if typ.lower() in ['hp', 'highpass']:
        return filc
    elif typ.lower() in ['lp', 'lowpass']:
        return 1 - filc


def fshift(w, s, axis=-1):
    """
    Shifts a 1D or 2D signal in frequency domain, to allow for accurate non-integer shifts
    :param w: input signal
    :param s: shift in samples, positive shifts forward
    :param axis: axis along which to shift (last axis by default)
    :return: w
    """
    # create a vector that contains a 1 sample shift on the axis
    ns = np.array(w.shape) * 0 + 1
    ns[axis] = w.shape[axis]
    dephas = np.zeros(ns)
    np.put(dephas, 1, 1)
    # fft the data along the axis and the dephas
    W = freduce(scipy.fft.fft(w, axis=axis), axis=axis)
    dephas = freduce(scipy.fft.fft(dephas, axis=axis), axis=axis)
    # if multiple shifts, broadcast along the other dimensions, otherwise keep a single vector
    if not np.isscalar(s):
        s_shape = np.array(w.shape)
        s_shape[axis] = 1
        s = s.reshape(s_shape)
    # apply the shift (s) to the fft angle to get the phase shift
    dephas = np.exp(1j * np.angle(dephas) * s)
    # apply phase shift by broadcasting
    out = np.real(scipy.fft.ifft(fexpand(W * dephas, ns[axis], axis=axis), axis=axis))
    return out.astype(w.dtype)


def fit_phase(w, si=1, fmin=0, fmax=None, axis=-1):
    """
    Performs a linear regression on the unwrapped phase of a wavelet to obtain a time-delay
    :param w: wavelet (usually a cross-correlation)
    :param si: sampling interval
    :param fmin: sampling interval
    :param fnax: sampling interval
    :param axis:
    :return: dt
    """
    if fmax is None:
        fmax = 1 / si / 2
    ns = w.shape[axis]
    freqs = freduce(fscale(ns, si=si))
    phi = np.unwrap(np.angle(freduce(np.fft.fft(w, axis=axis), axis=axis)))
    indf = np.logical_and(fmin < freqs, freqs < fmax)
    dt = - np.polyfit(freqs[indf],
                      np.swapaxes(phi.compress(indf, axis=axis), axis, 0), 1)[0] / np.pi / 2
    return dt


def dft(x, xscale=None, axis=-1, kscale=None):
    """
    1D discrete fourier transform. Vectorized.
    :param x: 1D numpy array to be transformed
    :param xscale: time or spatial index of each sample
    :param axis: for multidimensional arrays, axis along which the ft is computed
    :param kscale: (optional) fourier coefficient. All if complex input, positive if real
    :return: 1D complex numpy array
    """
    ns = x.shape[axis]
    if xscale is None:
        xscale = np.arange(ns)
    if kscale is None:
        nk = ns if np.any(np.iscomplex(x)) else np.ceil((ns + 1) / 2)
        kscale = np.arange(nk)
    else:
        nk = kscale.size
    if axis != 0:
        # the axis of the transform always needs to be the first
        x = np.swapaxes(x, axis, 0)
    shape = np.array(x.shape)
    x = np.reshape(x, (ns, int(np.prod(x.shape) / ns)))
    # compute fourier coefficients
    exp = np.exp(- 1j * 2 * np.pi / ns * xscale * kscale[:, np.newaxis])
    X = np.matmul(exp, x)
    shape[0] = int(nk)
    X = X.reshape(shape)
    if axis != 0:
        X = np.swapaxes(X, axis, 0)
    return X


def dft2(x, r, c, nk, nl):
    """
    Irregularly sampled 2D dft by projecting into sines/cosines. Vectorized.
    :param x: vector or 2d matrix of shape (nrc, nt)
    :param r: vector (nrc) of normalized positions along the k dimension (axis 0)
    :param c: vector (nrc) of normalized positions along the l dimension (axis 1)
    :param nk: output size along axis 0
    :param nl: output size along axis 1
    :return: Matrix X (nk, nl, nt)
    """
    # it would be interesting to compare performance with numba straight loops (easier to write)
    # GPU/C implementation should implement straight loops
    nt = x.shape[-1]
    k, h = [v.flatten() for v in np.meshgrid(np.arange(nk), np.arange(nl), indexing='ij')]
    # exp has dimension (kh, rc)
    exp = np.exp(- 1j * 2 * np.pi * (r[np.newaxis] * k[:, np.newaxis] +
                                     c[np.newaxis] * h[:, np.newaxis]))
    return np.matmul(exp, x).reshape((nk, nl, nt))

def trace_header(version=1):
    """
    For the dense layout used at IBL, returns a dictionary with keys
    x, y, row, col, ind, adc and sampleshift vectors corresponding to each site
    """
    h = dense_layout()
    h['sample_shift'], h['adc'] = adc_shifts(version=version)
    return h

def dense_layout():
    """Dictionary containing local coordinates of a Neuropixel 3 dense layout"""
    ch = {'ind': np.arange(NC),
          'col': np.tile(np.array([2, 0, 3, 1]), int(NC / 4)),
          'row': np.floor(np.arange(NC) / 2)}
    ch.update(rc2xy(ch['row'], ch['col']))
    return ch

def rc2xy(row, col):
    "converts the row/col indices from "
    x = col * 16 + 11
    y = (row * 20) + 20
    return {'x': x, 'y': y}

def adc_shifts(version=1):
    """
    The sampling is serial within the same ADC, but it happens at the same time in all ADCs.
    The ADC to channel mapping is done per odd and even channels:
    ADC1: ch1, ch3, ch5, ch7...
    ADC2: ch2, ch4, ch6....
    ADC3: ch33, ch35, ch37...
    ADC4: ch34, ch36, ch38...
    Therefore, channels 1, 2, 33, 34 get sample at the same time. I hope this is more or
    less clear. In 1.0, it is similar, but there we have 32 ADC that sample each 12 channels."
    - Nick on Slack after talking to Carolina - ;-)
    """
    if version == 1:
        adc_channels = 12
        # version 1 uses 32 ADC that sample 12 channels each
    elif version == 2:
        # version 2 uses 24 ADC that sample 16 channels each
        adc_channels = 16
    adc = np.floor(np.arange(NC) / (adc_channels * 2)) * 2 + np.mod(np.arange(NC), 2)
    sample_shift = np.zeros_like(adc)
    for a in adc:
        sample_shift[adc == a] = np.arange(adc_channels) / adc_channels
    return sample_shift, adc