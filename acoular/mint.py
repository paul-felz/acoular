import warnings
from .ism import Ism
from .signals import SignalGenerator
from .sources import MaskedTimeSamples
from traits.api import Trait, Property, Int, Str, Long, Array, List, Tuple, \
        Delegate, Float, on_trait_change, property_depends_on, HasPrivateTraits
from resampy import resample
from numpy import insert, zeros, append, column_stack, log10, argmax, \
        linspace, exp, flip, flipud, nonzero, array, where
from numpy.linalg import inv
from numpy.fft import fft, fftshift, ifft
from scipy.signal import convolve, fftconvolve 
from scipy.io import wavfile

class LoadSignal( SignalGenerator ):
    """
    Load signal generator. 
    """

    wavpath = Trait(Str) 

    data = Property()
    
    #start stop of audio in s
    #both get calculated if bigger than 0.0s
    starts = Int(0,
            desc="start time in samples")
    stops= Int(0,
            desc="stop time in samples")

    _numsamples = Long

    numsamples = Property()

    @on_trait_change('wavpath,starts,stops')
    def _get_data(self):
        fs, data = wavfile.read(self.wavpath)
        if data.ndim >1:
            data = data[:,0]
        if self.starts != 0:
            data = data[self.starts:]
        if self.stops != 0.0:
            stops = self.stops-self.starts
            data = data[:stops]
        if fs != self.sample_freq:
            data = resample(data, fs, self.sample_freq, axis=-1)
            numsamples = len(data)
            self.numsamples = numsamples
            warnings.warn('Warning: Resample of Sound file. Reconsider sample frequency change!')
            return data
        else:
            return data

    def _get_numsamples(self):
        return self._numsamples

    def _set_numsamples(self,numsamples):
        self._numsamples = numsamples
        
    def signal(self):
        """
        Deliver *.wav signal.


        Returns
        -------
        Array of floats
            The resulting signal as an array of length :attr:`~SignalGenerator.numsamples`.
        """
        return self.data

    def usignal(self, factor):
        """
        Delivers the signal resampled with a multiple of the sampling freq.
        
        Parameters
        ----------
        factor : integer
            The factor defines how many times the new sampling frequency is
            larger than :attr:`sample_freq`.
        
        Returns
        -------
        array of floats
            The resulting signal of length `factor` * :attr:`numsamples`.
        """
        return resample(self.signal(), self.sample_freq, factor*self.sample_freq)

class FiniteImpulseResponse(HasPrivateTraits):

    h = Property()
"""
class FiniteImpulseResponseMeasurement(FiniteImpulseResponse):
    #normalized impulse response 
    h = Property()

    def _get_h(self):
        #normalized length of impulse responses with hframe
        h = []
        for i in range (0,self.impulse_response.shape[1]):
            hchannel = array(self.impulse_response[self.hframe[i][0]:self.hframe[i][-1]+1,i])
            h.insert(i,hchannel)
        return h

"""
class FiniteImpulseResponseMeas(FiniteImpulseResponse):
    up = Int()

    impulse_response1 = Array()
    impulse_response2 = Array()

    measurement = List()

    h = Property()
    def _get_h(self):
        h = []
        h.insert(0,self.impulse_response1)
        h.insert(1,self.impulse_response2)
        return h

    def result(self):
        res = []
        restemp = self.measurement[0]
        res.insert(0,restemp)
        restemp = self.measurement[1]
        res.insert(1,restemp)
        yield res

    

class FiniteImpulseResponseSimulation(FiniteImpulseResponse):

    ism = Trait(Ism(),Ism)

    #loc = Delegate('ism','loc')
    loc = Tuple((0.0, 0.0, 1.0),
        desc="source location")

    up = Delegate('ism','up')

    #simulated impulse response
    impulse_response = Property()

    @property_depends_on('loc,ism')
    def _get_impulse_response(self):
        return self.ism.impulse_response(self.loc)

    #frame indices of valid impulse response values
    hframe = Property()

    @property_depends_on('impulse_response')
    def _get_hframe(self):
        hframe = []
        for i in range(0,self.impulse_response.shape[1]):
            hframechannel = where(self.impulse_response[:,i]!=0)
            hframe.insert(i,hframechannel[0])
        return hframe
    
    #impulse reponses array for each microfon input
    harray = Property()
    #normalized impulse response 
    h = Property()

    def choose_channel(self,h):
        #choose two impulse responses from microfon array that don't have the same zeros
        hlenchannel1 = len(h)
        hlenchannel2 = hlenchannel1
        while hlenchannel1:
            hlenchannel1 -=1
            while hlenchannel2:
                hlenchannel2 -= 1
                if len(h[hlenchannel1]) != len(h[hlenchannel2]):
                    break
            else:
                hlenchannel2 = len(h)
                continue
            break
        if len(h[hlenchannel1]) != len(h[hlenchannel2]):
            return hlenchannel1, hlenchannel2
        else:
            raise Exception("To invert the FIR system two impulse responses with different zeros are mandatory") 

    @property_depends_on('impulse_response,hframe')
    def _get_harray(self):
        #list with impulse responses of each available microfon channel
        harray = []
        for i in range (0,self.impulse_response.shape[1]):
            hchannel = array(self.impulse_response[self.hframe[i][0]:self.hframe[i][-1]+1,i])
            harray.insert(i,hchannel)
        return harray

    @property_depends_on('harray')
    def _get_h(self):
        #list with impulse responses of 2 chosen microfon channels
        h = []
        [hchannel1, hchannel2] = self.choose_channel(self.harray)
        print("Channel ",hchannel1," and channel ",hchannel2," selected for multiple input multiple output inverse filtering.")
        h.insert(0,self.harray[hchannel1])
        h.insert(1,self.harray[hchannel2])
        return h

    def result(self):
        hframe = self.hframe
        [hchannel1, hchannel2] = self.choose_channel(self.harray) 
        hframe1 = hframe[hchannel1]
        hframe2 = hframe[hchannel2]

        for item in self.ism.result(self.ism.numsamples):
            res = []
            restemp = item[hframe1[0]:,hchannel1]
            res.insert(0,restemp)
            restemp = item[hframe2[0]:,hchannel2]
            res.insert(1,restemp)
            yield res

class Mint(HasPrivateTraits):

    fir = Trait(FiniteImpulseResponse(),FiniteImpulseResponse)

    g = Property()
    
    @property_depends_on('fir')
    def _get_g(self):
        [h1,h2] = self.fir.h

        #Duration of impulse Responses channel 1 and 2
        m = len(h1)-1
        n = len(h2)-1

        #number of filter coefficients for inverse filtering channel 1 and 2
        i = n-1
        j = m-1

        #help to calc shape of output l = m+i = n+j
        l = m+i

        # d = h * g -- result of inverse filter convolution
        d= zeros((l))
        d= append(1,d)

        #mint impulse response matrices
        gtemp1 = zeros(i)
        gtemp1 = append(h1,gtemp1)
        g1 = gtemp1
        gtemp2 = zeros(j)
        gtemp2 = append(h2,gtemp2)
        g2 = gtemp2

        ind = i
        while ind:
            gtemp1 = append(gtemp1[-1:],gtemp1[:-1])
            g1 = column_stack((g1,gtemp1))
            ind-=1

        jnd = j
        while jnd:
            gtemp2 = append(gtemp2[-1:],gtemp2[:-1])
            g2 = column_stack((g2,gtemp2))
            jnd-=1

        #[hfilt1, hfilt2]^T = [g1, g2]^-1 * d
        gsquare = column_stack((g1,g2))
        gsquareinv = inv(gsquare)
        hfilt = gsquareinv*d

        hfilt1 = hfilt[0:i+1,0]
        hfilt2 = hfilt[i+1:i+2+j,0]
        g = [hfilt1, hfilt2]
        return g

    yrecovered = Property()
    
    @property_depends_on('fir,g')
    def _get_yrecovered(self):
        [hfilt1,hfilt2] = self.g
        for item in self.fir.result():
            yrecovered = convolve(hfilt1,item[0])
            yrecovered = yrecovered[::self.fir.up]
            yrecovered2 = convolve(hfilt2,item[1])
            yrecovered2 = yrecovered2[::self.fir.up]

        if len(yrecovered)>len(yrecovered2):
            lendiff = len(yrecovered)-len(yrecovered2)
            padding = zeros(lendiff)
            yrecovered2 = append(yrecovered2,padding)
        elif len(yrecovered)<len(yrecovered2):
            lendiff = len(yrecovered2)-len(yrecovered)
            padding = zeros(lendiff)
            yrecovered = append(yrecovered,padding)

        yrecov = yrecovered + yrecovered2
        return yrecov

class EvaluateMint(HasPrivateTraits):
    
    ts = Trait(MaskedTimeSamples(),MaskedTimeSamples)

    #lsignal = Instance(LoadSignal(), LoadSignal)
    sample_freq = Float()

    def ir_ess(self,x,channel):
        #get data
        #TODO: i = next(self.ts.result(self.ts.numsamples_total))
        for i in self.ts.result(self.ts.numsamples_total):
            #read and normalize
            ichannel = i[:,channel]
            ichannelmax = max(ichannel)
            ichannel = 1.0*(ichannel/ichannelmax)
            #cut out
            iabs = abs(ichannel)
            ilevel = ma.log10(iabs)
            ilevel = ilevel.filled(-15)
            ilevel *= 20
            indices = nonzero(ilevel>-30)
            y = ichannel[indices[0][0]:indices[0][-1]]


        #INVERSE FILTER
        xabs = abs(x)
        xlevel = ma.log10(xabs)
        xlevel = xlevel.filled(-15)
        xlevel *= 20
        indices = nonzero(xlevel>-30)
        x = x[indices[0][0]:indices[0][-1]]
        xinv = flip(x)
        
        #VERSION1
        lendiff = len(xinv)-len(y)
        if lendiff>0:
            zerodiff = zeros(lendiff)
            y = append(y,zerodiff)
        else:
            zerodiff = zeros(-lendiff)
            xinv = append(xinv,zerodiff)

        #Kompensation
        T = len(xinv)/self.sample_freq
        t = linspace(0,T,len(xinv))
        komp = 1/exp((log(20000/80)*t)/T)
        xinv = xinv*komp

        frp = fft(fftconvolve(xinv,y))
        xinv = xinv/abs(frp[round(frp.shape[0]/4)]) 
        #h = fftconvolve(y,xinv,mode='full') 

        #anti aliasing
        xinv = append(xinv,zeros(len(xinv)))
        y = append(y,zeros(len(y)))
        
        """
        nyq = 0.5*51200
        low = 100/nyq
        high= 18000/nyq
        b,a = butter(5, [low,high], btype='band')
        y = lfilter(b,a,y)
        xinv = lfilter(b,a,xinv)
        """

        h = convolve(xinv,y)

        indmax = argmax(abs(h))
        h = h[indmax:]     
        h = h[100:]
        indmax = argmax(abs(h))
        h = h[indmax:]
        hmax = max(abs(h))
        h = h/hmax
        return h
     
    """
    def correlate_time_series(record1,record2):
        #normalize and cut
        #record1max = max(record1)                       
        #record1 = 0.95*(record1/record1max)                                 
        record1abs = abs(record1)                                                  
        record1level = ma.log10(record1abs)                                        
        record1level = record1level.filled(-15)                                    
        record1level *= 20                                                         
        indices1 = nonzero(record1level>-30)                                       
        record1 = record1[indices1[0][0]:indices1[0][-1]] 
        
        #record2max = max(record2)                                               
        #record2 = 0.95*(record2/record2max)                                 
        record2abs = abs(record2)                                               
        record2level = ma.log10(record2abs)                                     
        record2level = record2level.filled(-15)                                 
        record2level *= 20                                                  
        indices2 = nonzero(record2level>-30)                                
        record2 = record2[indices2[0][0]:indices2[0][-1]] 

        lendiff = len(record1)-len(record2)                                     
        padding = zeros(abs(lendiff))                                           
        if lendiff>0:                                                       
            record2 = append(record2,padding)                               
        elif lendiff<0:                                                     
            record1 = append(record1,padding)
            
        record1 = (record1 - mean(record1)) / (std(record1) * len(record1))
        record2 = (record2 - mean(record2)) / (std(record2))
        co = correlate(record1,record2,"full")
        return max(co)

    def correlate_series(record1,record2):
        a = record1
        b = record2
        #b = zeros(data_length * 2)

        #breakpoint()
        #b[data_length/2:data_length/2+data_length] = record2 # This works for data_length being even

        # Do an array flipped convolution, which is a correlation.
        c = convolve(b, a[::-1], mode='full') 
        return max(c)
    """
    def cross_correlation_using_fft(x, y):
        f1 = fft(x)
        f2 = fft(flipud(y))
        cc = real(ifft(f1 * f2))
        return fftshift(cc)

    # shift < 0 means that y starts 'shift' time steps before x # shift > 0 means that y starts 'shift' time steps after x
    def compute_shift(x, y):
        assert len(x) == len(y)
        c = EvaluateMint.cross_correlation_using_fft(x, y)
        assert len(c) == len(x)
        #zero_index = int(len(x) / 2) - 1
        zero_index = int(len(x) / 2)-1
        shift = zero_index - argmax(c)
        return shift

    def cut_around(record):
        record1abs = abs(record)                                                  
        record1level = ma.log10(record1abs)                                        
        record1level = record1level.filled(-15)                                    
        record1level *= 20                                                         
        indices1 = nonzero(record1level>-30)                                       
        record = record[indices1[0][0]:indices1[0][-1]] 
        return record

    def fit_length(record1,record2):
        #fit sizes of recordings
        lendiff = len(record1)-len(record2)                                                                                                                        
        padding = zeros(abs(lendiff))                                                                                                                            
        record1 = append(record1,padding)    
        return record1
