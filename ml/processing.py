__author__ = 'joren'

from functools import reduce
import math

import numpy as np
import pywt

#
from signals.primitive import Transformer, SignalBlock

def interp_vals(val1, val2, index, total):
    return val1 + ( (val2-val1) * (index/total) )

def interp_ls(l1, l2, i):
    return l1[i:] + l2[0:i]

def wavelet_trans(l):
    #TODO binning per decomposition level?
    #TODO explain why db4
    coeffs = pywt.wavedec(l, 'db4', level=pywt.dwt_max_level(len(l),pywt.Wavelet('db4')))
    return merge(*coeffs[0:len(coeffs)//2])

def merge(*args):
    return reduce(lambda a,b: np.append(a, b), args)

def fourier_trans(l, f_sample = 512, freq_low = 4, freq_high = 50):
    """
    The fft of input list l. As the Mindwave's sampling frequency is 512 hz that is the default for f_sample.
    Giving the sampling frequency allows us to accurately band-filter the result.
    This is necessary as frequencies under 4 or over 40 hz are generally not considered useful in brainwave data.
    :param l: List of measurements
    :param freq_low: Low end cut-off frequency
    :param freq_high: High end cut-off frequency
    :return: Filtered Fourier coefficients
    """
    #TODO binning? Must be based on experimental results! (?)
    ft = np.fft.rfft(l)
    i_s = math.floor(freq_low*f_sample/len(l))
    i_e = math.ceil(freq_high*f_sample/len(l))
    return np.absolute(ft[i_s:i_e])

def extremes(l):
    a = np.array(l)
    (mp,lmp) = a.max(), a.argmax()
    (mm,lmm) = (-a).max(), (-a).argmax()
    return np.array([mp, lmp, mm, lmm])

class Preprocessor(Transformer):
    def __init__(self, source, transform, extra_params={}):
        """
        :param source: The signal source to be preprocessed
        :param transform: A function that takes at least a list and only kwargs beyond that
        :param extra_params: Extra things the transform function needs
        """
        Transformer.__init__(self
                            ,[source]
                            ,{source.getName(): 'l'}
                            ,lambda l: transform(l,**extra_params))

class Preprocessing(SignalBlock):
    def __init__(self, source):
        self.wt = Preprocessor(source, wavelet_trans)
        self.ft = Preprocessor(source, fourier_trans)
        self.et = Preprocessor(source, extremes)
        merger = Transformer([self.wt, self.ft, self.et]
                            ,{self.wt.getName(): 'w',
                              self.ft.getName(): 'f',
                              self.et.getName(): 'e'}
                            ,merge)
        SignalBlock.__init__(self, [source], [merger])
