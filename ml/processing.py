__author__ = 'joren'

import math
from functools import reduce

import numpy as np
import pywt
#
from signals.primitive import Transformer


def wavelet_trans(l):
    #TODO binning per decomposition level?
    #TODO explain why db4
    if not (l is None or l == []):
        coeffs = pywt.wavedec(l, 'db4', level=pywt.dwt_max_level(len(l),pywt.Wavelet('db4')))
        return merge(*coeffs[0:len(coeffs)//2])
    else:
        return l

def merge(*args):
    if args:
        input = list(args)
        return reduce(lambda a,b: np.append(a, b), input)
    else:
        return []

def fix_length(l, length):
    if len(l) == length:
        return l
    elif len(l) > length:
        return l[0:length+1]
    else:
        print l
        return l + [0 for i in xrange(length - len(l))]

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
    if not (l is None or l == []):
        #TODO binning? Must be based on experimental results! (?)
        ft = np.fft.rfft(l)
        i_s = math.floor(freq_low*f_sample/len(l))
        i_e = math.ceil(freq_high*f_sample/len(l))
        return np.absolute(ft[i_s:i_e])
    else:
        return l

def extremes(l):
    if not (l is None or l == []):
        a = np.array(l)
        (mp,lmp) = a.max(), a.argmax()
        (mm,lmm) = (-a).max(), (-a).argmax()
        return np.array([mp, lmp, mm, lmm])
    else:
        return l

class Preprocessor(Transformer):
    def __init__(self, source, transform, extra_params=None):
        """
        :param source: The signal source to be preprocessed
        :param transform: A function that takes at least a list and only kwargs beyond that
        :param extra_params: Extra things the transform function needs
        """
        params = extra_params or {} # Mutable kwargs 'n all that
        Transformer.__init__(self
                            ,[source]
                            ,{source.getName(): 'l'}
                            ,lambda l: transform(l,**params))

class Preprocessing(Transformer):
    def __init__(self, source):
        self.original_source = source
        if hasattr(source, 'select'):
            raw_source = source.select('raw')
        else:
            raw_source = source
        fixlentrans = Transformer([raw_source], {raw_source.getName(): 'l'}, lambda l: fix_length(l, 512))
        self.wt = Preprocessor(fixlentrans, wavelet_trans)
        self.ft = Preprocessor(fixlentrans, fourier_trans)
        self.et = Preprocessor(fixlentrans, extremes)
        Transformer.__init__(self,
                            [self.wt, self.ft, self.et]
                            ,{self.wt.getName(): 'w',
                              self.ft.getName(): 'f',
                              self.et.getName(): 'e'}
                            ,lambda w,f,e: merge(w,f,e))
        #if hasattr(source, 'initialized'):
        #    if not source.initialized:
        #        source.push()
        source.push()
        self.pull()
        # For some reason this needs to happen at least once for the output_dim to be correct
        # TODO check why ^
        self.output_dim = len(self.pull())

    def getName(self):
        return "-prep-"
