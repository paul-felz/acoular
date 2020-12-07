import warnings
import h5py
from numpy import transpose, array, zeros, concatenate, delete, where, floor, dot, subtract, \
pi, complex128, float32, sin, cos, isscalar, cross, sqrt, absolute, einsum, newaxis, \
ndarray, rint, empty, int64, ones, append, floor, log10, \
nonzero, log, ma, argmax, mean, std, real, sinc, ceil, \
arange, divide
from numpy.linalg import norm
from scipy.signal import convolve

from traits.api import HasTraits, HasPrivateTraits, Float, \
Property, Instance, Trait, Bool, Range, Delegate, Enum, Any, \
cached_property, on_trait_change, property_depends_on, List, Tuple

from .internal import digest
from .sources import SamplesGenerator, SourceMixer, PointSource
from .grids import Grid, RectGrid
from .environments import Environment
from .microphones import MicGeom
from .fbeamform import SteeringVector
from .signals import SignalGenerator
from .trajectory import Trajectory

from .fastFuncs import calcTransfer

from enum import Enum
import pdb

class Orientation(Enum):
    """
    This represents the orientation of a wall on the xyz plane.
    This class is useful to create walls perpendicular to x-, y- and z-axis.
    X = Enum(0)
    Y = Enum(1)
    Z = Enum(2)
    """
    X = 0
    Y = 1
    Z = 2

class Wall(HasPrivateTraits):
    """
    Base class for general wall in Hesse normal form 
    """
    alpha = Float(0.0,
            desc="absorption coefficient")

    n0 = Property()
    
    @property_depends_on('n0')
    def _get_n0(self):
        return self.n0

class WallHesse(Wall):
    """
    Returns the hesse normal form of a wall plane.
    """
    point1 = Tuple((0.0,0.0,0.0),
            desc="point on plane")
    
    n0 = Tuple((1.0,0.0,0.0),
            desc="normal vector of plane")

    
    @property_depends_on('point1')
    def _get_point1(self):
        return self.point1

    @property_depends_on('n0')
    def _get_n0(self):
        return self.n0


class Wall3Points(Wall):
    """
    Calculates the Hesse normal form of a wall plane with 3 given points.
    """
    point1 = Tuple((0.0,0.0,0.0),
            desc="point 1 on plane")
    point2 = Tuple((0.0,0.0,0.0),
            desc="point 2 on plane")
    point3 = Tuple((0.0,0.0,0.0),
            desc="point 3 on plane")
    
    u = Property()
    v = Property()
    #n0 = Property()

    @property_depends_on('point1,point2')
    def _get_u(self):
        return tuple(subtract(self.point2,self.point1))

    @property_depends_on('point1,point3')
    def _get_v(self):
        return tuple(subtract(self.point3,self.point1))

    @property_depends_on('u, v')
    def _get_n0(self):
        n = cross(self.u,self.v)
        self.check_valid_points(n)
        return tuple(n/norm(n))

    def check_valid_points(self,n):
        """
        Check if points are valid to span a specific plane in R^3.
        """
        if all(i == 0 for i in n):
            print("Points are the same or in one line. This leads to an ambiguous assignement of a plane.")
            raise ValueError


class WallOrientation(Wall):
    """
    Calculates the hesse normal form of a wall plane perpendicular to x-, y- or z-axis.
    """
    position = Float(0.0,
            desc="describes position of wall plane on corresponding axis described in orientation")

    orientation = Trait(Orientation,
            desc="describes axis, that is perpendicular to wall plane")

    point1 = Property(desc="base point of wall")
    
    @property_depends_on('position, orientation')
    def _get_point1(self):
        if self.orientation == Orientation.X:
            point1 = (self.position,0.0,0.0)
        elif self.orientation == Orientation.Y:
            point1 = (0.0, self.position, 0.0)
        else:
            point1 = (0.0, 0.0, self.position)
        return point1
    
    @property_depends_on('position, orientation')
    def _get_n0(self):
        if self.orientation == Orientation.X:
            n0 = (1.0,0.0,0.0)
        elif self.orientation == Orientation.Y:
            n0 = (0.0,1.0,0.0)
        else:
            n0 = (0.0,0.0,1.0)
        return n0

class Room(Environment):
    """
    Turns the simple acoustic environment into a more advanced environment of a room.
    For example: Single wall, a cuboid.
    """
    walls = List( Instance(Wall,()) ,
            desc="List of Instances wall which contents all wall planes of a specific room.")

    def add_wall(self, wall):
        self.walls.append(wall)

    def create_wall_3points(self, point1, point2, point3, alpha):
        wall = Wall3Points(point1=point1,point2=point2,point3=point3,alpha=alpha)
        self.add_wall(wall)
        return self

    def create_wall_orientation(self, position, orientation,alpha):
        wall = WallOrientation(position=position, orientation=orientation,alpha=alpha)
        self.add_wall(wall)
        return self
    
    def create_wall_hesse(self, point1, n0, alpha):
        wall = WallHesse(point1=point1, n0=n0, alpha=alpha)
        self.add_wall(wall)
        return self

class IsmRealImages(SamplesGenerator):
    """
    Image source model for fast beamforming.
    Mirrors the point source "pyhsicaly" behind walls of room and writes them to the list sources.
    J. Allen and D. Berkeley, "Image method for efficiently simulating small-room acoustics"
    """
    source = Instance(SamplesGenerator(),SamplesGenerator,
            desc="source that gets mirrored by Ism")

    room = Trait(Room,
            desc="room with list of wall planes")

    sources = Property(desc="List of mirrored sources.")

    sample_freq = Delegate('source','sample_freq') 

    numchannels = Delegate('source','numchannels')

    numsamples = Delegate('source','numsamples')

    def _get_sources(self):
        sources = []
        sources.extend([self.source])
        for wall in self.room.walls:
            temp = self.clone_traits()
            temp.source.loc = temp.mirror_loc(wall.n0,wall.point1)
            sources.extend([temp.source])
        return sources

                    
    def mirror_loc(self, n0, point1):
        """
        Reflects and returns the location of a source on the backside of a wall.

        Parameters
        ----------
        n0: normal vector of wall plane
        point1: base point of wall plane

        Returns
        -------
        tuple of reflected loc
        """
        d = dot(self.source.loc,n0) - dot(point1,n0)
        n02d = tuple(2*d*x for x in n0)
        mirror_loc = subtract(self.source.loc,n02d)
        mirror_loc = tuple(mirror_loc)
        return mirror_loc    

    def result(self,num):
        """
        Python generator that yields the output block-wise.
        The outputs from the reflected sources in the list are being added.
        Parameters
        ----------
        num: integer
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block).
        Returns
        -------
        Samples in blocks of shape (num, numchannels).
            The last block may shorter than num.
        """
        gens = [i.result(num) for i in self.sources[1:]]
        for temp in self.sources[0].result(num):
            sh = temp.shape[0]
            ind = 0
            for g in gens:
                temp1 = next(g)
                beta = sqrt(1-self.room.walls[ind].alpha)
                temp += beta * temp1
                ind += 1
            yield temp
            if sh > temp.shape[0]:
                break


class Ism(PointSource):
    """
    Image source model for beamforming with signal treatment.
    In contrast to the IsmRealImages class, the sources doesn't get mirrored "physicaly". 
    Instead, there is just one source signal convoluted with the impulse response
    from the Ism. This way there is more signal treatment possible and the beamforming happens
    just on one treated source.
    J. Allen and D. Berkeley, "Image method for efficiently simulating small-room acoustics"
    """
    room = Trait(Room,
            desc="room with list of wall planes")

    def plane_distance(self,point,planepoint,n0):
        """
        Calculates the distance from a point to a plane in hesse normal form.
        The calculated distance is with regard to the orientation of n0 positive or negative.
        """
        return dot(point,n0)-dot(planepoint,n0)

    def mirror_loc(self,loc, n0, point1):
        """
        Reflects and returns the location of a source on the backside of a wall.

        Parameters
        ----------
        n0: normal vector of wall plane
        point1: base point of wall plane

        Returns
        -------
        tuple of reflected loc
        """
        d = self.plane_distance(loc,point1,n0)
        n02d = tuple(2*d*x for x in n0)
        mirror_loc = subtract(loc,n02d)
        mirror_loc = tuple(mirror_loc)
        return mirror_loc    
    
    def impulse_response(self):
        pass

    def result(self,num):
        pass

class FilterRIR(Enum):
    """
    Enum class to choose the discrete delta LPF{delta(n- tau*fs)}=
    round: delta(n-round(tau*fs))
    hanningLPF:  = 1/2 * (1+ cos(2*pi*t/tw))sinc(2pi*fc*t) for -tw/2<t<tw/2
                    0                                           otherwise
    source: Habets, Emanuel A.P.. Room Impulse Response Generator. 2017
    """
    RoundLpf = 0
    HanningLpf = 1


class PointSourceIsm(Ism):
    """
    Class to define a fixed point source in a reverberant environment.
    This can be used for beamforming a simulated or loaded signal in a simulated room.
    
    The output is being generated via the :meth:`result` generator.
    """
    filterrir = Trait(FilterRIR,
            desc="Filter used for the compensation of the impulse response discrete estimation error.")
    
    #: Number of samples, is set automatically / 
    #: depends on :attr:`signal`.
    numsamples = Property(depends_on = ['loc','signal'])

    def _get_numsamples(self):
        return self.signal.numsamples*self.up+self.impulse_response(self.loc).shape[0]-1

    def impulse_response(self,loc):
        """
        Calculates impulse response of a given source location in relation to a given room.
        """
        #hdirect
        h = self.calc_h(loc)
        hlen = h.shape[0]
        #add reflexions to h and append size of h with longest reflexion size
        for wall in self.room.walls:
            locm = self.mirror_loc(loc,wall.n0,wall.point1)
            alpha = wall.alpha
            beta = sqrt(1-alpha)
            hreflexion = beta * self.calc_h(locm)
            #adapt size to longest hreflexion
            if hlen<hreflexion.shape[0]:
                dim1 = hreflexion.shape[0]-hlen
                ext = zeros((dim1,self.numchannels))
                h = append(h,ext,0)
                hlen = h.shape[0]
            else:
                dim1 = hlen-hreflexion.shape[0]
                ext = zeros((dim1,self.numchannels))
                hreflexion = append(hreflexion,ext,0)
                hlen = hreflexion.shape[0]
            h += hreflexion
        return h

    def hanning_filt(self,t,tw,fc):
        """
        Calculates the hanning window filter function to replace the delta impulse of h.
        """
        hanningfilt = 1/2*(1+cos(2*pi*t/tw))*sinc(2*pi*fc*t)
        return hanningfilt

    def calc_h(self,loc):
        """
        Calculates h matrix for a given position in relation to microphones.
        """
        #travel distance
        rm = self.env._r(array(loc).reshape((3,1)), self.mics.mpos)
        #travel time index
        ind = (rm/self.env.c)*self.sample_freq*self.up
        #distance factor
        amp = 1/rm
        #What filter should be applied to delta impulse
        if self.filterrir == FilterRIR.HanningLpf:
            #discrete impulse response preparation
            twhalf = 0.002 #twhalf = 2ms Peterson 1986
            twhalfsamples = int(ceil(twhalf*self.sample_freq*self.up))
            #future len of h
            ind_max = rint(ind).max()
            ind_max = ind_max.astype(int)+twhalfsamples
            ind = array(0.5+ind,dtype=int64)
            #preparation of arrays
            h = zeros((ind_max+1, self.numchannels))
            t = arange(-twhalfsamples,twhalfsamples,1)/(self.sample_freq*self.up)
            if ind.size == 1:
                #h[ind[0],0] = amp[0]
                if ind[0]-twhalfsamples > 0:          #start with 0 
                    start_impulse=-twhalfsamples+ind[0]
                    tind = 0
                else:
                    start_impulse=0
                    tind = abs(ind[0]-twhalfsamples)
                for j in range(start_impulse,ind[0]+twhalfsamples):  #hanning func -tw/2<t<tw/2
                    hanning = self.hanning_filt(t[tind],2*twhalf,self.sample_freq/2)
                    h[j,0]=amp[0]*hanning
                    tind+=1
                #TODO: Unittest single impulse
            else:
                for i in range(0,ind.size):
                    if ind[0,i]-twhalfsamples > 0:          #start with 0 
                        start_impulse=-twhalfsamples+ind[0,i]
                        tind = 0
                    else:
                        start_impulse=0
                        tind = abs(ind[0,i]-twhalfsamples)
                    for j in range(start_impulse,ind[0,i]+twhalfsamples):  #hanning func -tw/2<t<tw/2
                        hanning = self.hanning_filt(t[tind],2*twhalf,(self.sample_freq)/2)
                        h[j,i]=amp[0,i]*hanning
                        tind +=1
        elif self.filterrir == FilterRIR.RoundLpf:
            ind_max = rint(ind).max()
            ind_max = ind_max.astype(int)
            ind = array(0.5+ind,dtype=int64)
            #preparation of array
            h = zeros((ind_max+1, self.numchannels))
            if ind.size == 1:
                h[ind[0],0] = amp[0]
            else:
                for i in range(0,ind.size):
                    h[ind[0,i],i] = amp[0,i]
        return h

    def result(self, num=128):
        """
        Python generator that yields the output at microphones block-wise.
                
        Parameters
        ----------
        num : integer, defaults to 128
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block) .
        
        Returns
        -------
        Samples in blocks of shape (num, numchannels). 
            The last block may be shorter than num.
        """
        signal = self.signal.usignal(self.up)
        out = zeros((num, self.numchannels))
        h = self.impulse_response(self.loc)
        #y.shape = len(convolved_signal)*numchannels
        y = empty((self.numsamples,self.numchannels))
        #convolve signal with impulse response
        for j in range(0,self.numchannels):
            y[:,j] = convolve(signal,h[:,j])
        #in-block iterator
        i = 0
        ind = 0
        #start and stop declarations
        ts = self.start_t*self.sample_freq
        ts = rint(ts).astype(int)
        tm = self.start*self.sample_freq
        tm = rint(tm).astype(int)
        #length of downsampled and with impulse response convolved signal + start_t time
        n = y.shape[0]/self.up + ts
        while n:
            n -= 1
            if ts>0:
                if tm>0:
                    tm-=1
                else:
                    out[i]=0.0
                    i+=1
                    if i == num:
                        yield out
                        i = 0
                ts-=1
            else:
                try:
                    out[i] = y[(ind+tm)*self.up,:]
                    i += 1
                    ind += 1
                    if i == num:
                        yield out
                        i = 0
                except IndexError: #if no more samples available from the source
                    break
        if i > 0: # if there are still samples to yield
            yield out[:i]         

class MovingPointSourceIsm( PointSourceIsm ):
    """
    Class to define a moving point source along a given trajectory in a reverberant room.
    This can be used for beamforming a simulated or loaded signal in a simulated room.
    This can be used in simulations.
    
    The output is being generated via the :meth:`result` generator.
    """

    #: Trajectory of the source, 
    #: instance of the :class:`~acoular.trajectory.Trajectory` class.
    #: The start time is assumed to be the same as for the samples.
    trajectory = Trait(Trajectory, 
        desc="trajectory of the source")

    numsamples = Delegate("signal")

    # internal identifier
    digest = Property( 
        depends_on = ['mics.digest', 'signal.digest', 'loc', \
         'env.digest', 'start_t', 'start', 'trajectory.digest', '__class__'], 
        )
               
    @cached_property
    def _get_digest( self ):
        return digest(self)

    def mirror_trajectory(self,n0,point1):
        trajectory = self.trajectory.clone_traits()
        points = {}
        for key, loc in self.trajectory.points.items():
            d = self.plane_distance(loc,point1,n0)
            n02d = tuple(2*d*x for x in n0)
            mirror_loc = subtract(loc,n02d)
            mirror_loc = tuple(mirror_loc)
            points[key] = mirror_loc
        trajectory.points = points
        return trajectory

    def impulse_response(self,epslim,t):
        h = self.calc_h(self.trajectory,epslim,t)
        hlen = h.shape[0]
        for wall in self.room.walls:
            trajectory = self.mirror_trajectory(wall.n0,wall.point1)
            hreflexion = self.calc_h(trajectory,epslim,t)
            if hlen<hreflexion.shape[0]:
                dim1 = hreflexion.shape[0]-hlen
                ext = zeros((dim1,self.numchannels))
                h = append(h,ext,0)
                hlen = h.shape[0]
            else:
                dim1 = hlen-hreflexion.shape[0]
                ext = zeros((dim1,self.numchannels))
                hreflexion = append(hreflexion,ext,0)
                hlen = hreflexion.shape[0]
            h += hreflexion
        return h

    def calc_h(self,trajectory,epslim,t):
        #travel distance
        te, rm = self.delay_distance_movingsource(trajectory,epslim,t)
        #travel time
        ind2 = (rm/self.env.c)*self.sample_freq
        ind = abs(te-t)*self.sample_freq
        ind_max = rint(ind).max()
        ind_max = ind_max.astype(int)
        amp = 1/rm
        h = zeros((ind_max+1, self.numchannels))
        ind = array(0.5+ind,dtype=int64)
        if ind.size == 1:
            h[ind[0],0] = amp[0]
        else:
            for i in range(0,ind.size):
                h[ind[i],i] = amp[i]
        return h
        

    def impulse_response_direct(self,epslim,t):
        #travel distance
        te, rm = self.delay_distance_movingsource(self.trajectory,epslim,t)
        #travel time
        ind2 = (rm/self.env.c)*self.sample_freq
        ind = abs(te-t)*self.sample_freq
        ind_max = rint(ind).max()
        ind_max = ind_max.astype(int)
        amp = 1/rm
        h = zeros((ind_max+1, self.numchannels))
        ind = array(0.5+ind,dtype=int64)
        if ind.size == 1:
            h[ind[0],0] = amp[0]
        else:
            for i in range(0,ind.size):
                h[ind[i],i] = amp[i]
        return h
    
    def impulse_response_reflexion(self,walls):
        trajectories = []
        for wall in self.room.walls:
            temp = self.clone_traits()
            temp.source.loc = temp.mirror_loc(wall.n0,wall.point1)
            sources.extend([temp.source])
        return hreflexion
    
    def delay_distance_movingsource(self,trajectory,epslim,t):
        j = 0
        eps = ones(self.mics.num_mics)
        te = t.copy() # init emission time = receiving time
        loc = array(trajectory.location(te))
        rm = loc-self.mics.mpos# distance vectors to microphones
        rm = sqrt((rm*rm).sum(0))# absolute distance
        loc /= sqrt((loc*loc).sum(0))# distance unit vector
        # Newton-Rhapson iteration
        while abs(eps).max()>epslim and j<100:
            der = array(trajectory.location(te, der=1))
            Mr = (der*loc).sum(0)/self.env.c# radial Mach number
            eps = (te + rm/self.env.c-t)/(1+Mr)# discrepancy in time 
            te -= eps
            j += 1 #iteration count
            loc = array(trajectory.location(te))
            rm = loc-self.mics.mpos# distance vectors to microphones
            rm = sqrt((rm*rm).sum(0))# absolute distance
            loc /= sqrt((loc*loc).sum(0))# distance unit vector
        return te, rm

    def result(self, num=128):
        """
        Python generator that yields the output at microphones block-wise.
                
        Parameters
        ----------
        num : integer, defaults to 128
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block).
        
        Returns
        -------
        Samples in blocks of shape (num, numchannels). 
            The last block may be shorter than num.
        """   
        signal = self.signal.usignal(self.up)
        out = zeros((num, self.numchannels))
        # shortcuts and intial values
        t = self.start*ones(self.mics.num_mics)
        i = 0
        ind = 0
        epslim = 0.1/self.up/self.sample_freq
        k = 0
        convsignaln = self.numsamples
        y = zeros((convsignaln,self.numchannels))
        ytemp = zeros((convsignaln,self.numchannels))

        #determine length n of result function (either shorter signal length or shorter trajectory length)
        numtrajectory = list(self.trajectory.points.items())[-1][0]
        if numtrajectory==self.numsamples/self.sample_freq:
            n = self.numsamples
        elif numtrajectory>=self.numsamples/self.sample_freq:
            n = self.numsamples
            temp = n/self.sample_freq
            warnstring = "time of signal (t=%f s) is shorter than time of trajectory (t=%f s)!" %(temp,numtrajectory)
            warnings.warn(warnstring)
        else:
            n = numtrajectory*self.sample_freq
            temp = self.numsamples/self.sample_freq
            warnstring = "time of trajectory (t=%f s) is shorter than time of signal (t=%f s)!" % (numtrajectory, temp)
            warnings.warn(warnstring)
        while n:
            h = self.impulse_response(epslim,t)
            ylen = self.numsamples+h.shape[0]
            ylen = rint(ylen).astype(int)
            if k<self.numsamples:
                if ylen>convsignaln:
                    n+=ylen-convsignaln
                    dim1 = (ylen-convsignaln)
                    ext = zeros((dim1,self.numchannels))
                    y = append(y,ext, 0)
                    convsignaln = ylen
                ytemp = zeros((convsignaln,self.numchannels))
                
                for j in range(0,self.numchannels):
                    c = convolve([signal[ind*self.up]],h[:,j])
                    ytemp[ind:ind+c.shape[0],j]=c
                
                y = y + ytemp

            t += 1./self.sample_freq
            try:
                out[i] = y[array(ind),:]
                i += 1
                ind += 1
                if i == num:
                    print("n: ",n)
                    yield out
                    i = 0
            except StopIteration:
                return
            k += 1
            n -= 1
        if i>0:
            return out[:j]

class GridExtender(Grid):
    """
    Reflects grid on walls of room and writes them in list mirrgrids.
    """
    grid = Instance(Grid(), Grid)
    
    room = Trait(Room,
            desc="room walls")

    #mirrgrids represent list of arrays with on wall reflected grids
    mirrgrids = Property(
            desc="List of reflected grid points")

    gpos = Delegate('grid','gpos') 

    def mirror_gridpoint(self, n0, basepoint, point):
        d = dot(point,n0) - dot(basepoint,n0)
        n02d = tuple(2*d*x for x in n0)
        mirr_gridpoint = subtract(point,n02d)
        mirr_gridpoint = tuple(mirr_gridpoint)
        return mirr_gridpoint


    #@property_depends_on('grid, room')
    def _get_mirrgrids( self ):
        gpos = self.grid.pos()
        mirrgrids = []
        for wall in self.room.walls:
            mirr_gpos = zeros([3,gpos.shape[1]])
            for i in range(0,mirr_gpos.shape[1]):
                gridpoint = (gpos[0,i],gpos[1,i],gpos[2,i])
                gridpoint_mirr = self.mirror_gridpoint(wall.n0, wall.point1, gridpoint)
                mirr_gpos[0,i] = gridpoint_mirr[0]
                mirr_gpos[1,i] = gridpoint_mirr[1]
                mirr_gpos[2,i] = gridpoint_mirr[2]
            mirrgrids.extend([mirr_gpos])
        return mirrgrids

    @property_depends_on('_nxsteps,_nysteps')
    def _get_size(self):
        return self.grid.size

    @property_depends_on('_nxsteps, _nysteps')
    def _get_shape ( self ):
        return self.grid.shape
    
class SteeringVectorRoom( SteeringVector ):
    
    room = Instance(Room(),Room)

    r0mirror = Property(desc="array center to mirror grid distances")

    rmirror = Property(desc="List of rm for mirror grids")

    steer_vector = Property(desc="calculated Steer-Vector")
    
    def _get_r0mirror ( self ):
        if isinstance(self.grid,GridExtender):
            if isscalar(self.ref) and self.ref > 0:
                r0i = []
                for mirrgrid in self.grid.mirrgrids: 
                    r0temp = full((mirrgrid.size,), self.ref)
                    r0i.append(r0temp)
                return r0i
            else:
                r0i = []
                for mirrgrid in self.grid.mirrgrids:
                    r0temp = self.env._r(mirrgrid)
                    r0i.append(r0temp)
                return r0i
        else:
            raise(TraitError(args=self,
                            name='r0mirror',
                            info='SteeringVectorRoom',
                            value=r0mirror))

    def _get_rmirror ( self ):
        if isinstance(self.grid,GridExtender):
            rmi = []
            for mirrgrid in self.grid.mirrgrids:
                rmtemp = self.env._r(mirrgrid,self.mics.mpos)
                rmi.append(rmtemp)
            return rmi
        else:
            raise(TraitError(args=self,
                            name='rmirror',
                            info='SteeringVectorRoom',
                            value=rmirror))

    def transfer_mirror(self,transm):
        for mirrNum in range(len(self.grid.mirrgrids)):
            transm += (1-self.room.walls[mirrNum].alpha) * calcTransfer(self.r0mirror[mirrNum],self.rmirror[mirrNum],array(2*pi*f/self.env.c))
        return transm

    def transfer(self, f, ind=None):
        """
        Calculates the transfer matrix for one frequency. 
        
        Parameters
        ----------
        f   : float
            Frequency for which to calculate the transfer matrix
        ind : (optional) array of ints
            If set, only the transfer function of the gridpoints addressed by 
            the given indices will be calculated. Useful for algorithms like CLEAN-SC,
            where not the full transfer matrix is needed
        
        <Down>
        -------
        array of complex128
            array of shape (ngridpts, nmics) containing the transfer matrix for the given frequency
        """
        #if self.cached:
        #    warn('Caching of transfer function is not yet supported!', Warning)
        #    self.cached = False
        
        if ind is None:
            trans = calcTransfer(self.r0, self.rm, array(2*pi*f/self.env.c))
            for mirrNum in range(len(self.grid.mirrgrids)):
                trans += sqrt(1-self.room.walls[mirrNum].alpha) * calcTransfer(self.r0mirror[mirrNum],self.rmirror[mirrNum],array(2*pi*f/self.env.c))
        elif not isinstance(ind,ndarray):
            trans = calcTransfer(self.r0[ind], self.rm[ind, :][newaxis], array(2*pi*f/self.env.c))#[0, :]
            for mirrNum in range(len(self.grid.mirrgrids)):
                trans += sqrt(1-self.room.walls[mirrNum].alpha) * calcTransfer(self.r0mirror[mirrNum][ind],self.rmirror[mirrNum][ind,:][newaxis],array(2*pi*f/self.env.c))
        else:
            trans = calcTransfer(self.r0[ind], self.rm[ind, :], array(2*pi*f/self.env.c))
            for mirrNum in range(len(self.grid.mirrgrids)):
                trans += sqrt(1-self.room.walls[mirrNum].alpha) * calcTransfer(self.r0mirror[mirrNum][ind],self.rmirror[mirrNum][ind,:],array(2*pi*f/self.env.c))
        return trans

    def steer_vector(self, f, ind=None):
        """
        Calculates the steering vectors based on the transfer function
        See also :ref:`Sarradj, 2012<Sarradj2012>`.
        
        Parameters
        ----------
        f   : float
            Frequency for which to calculate the transfer matrix
        ind : (optional) array of ints
            If set, only the steering vectors of the gridpoints addressed by 
            the given indices will be calculated. Useful for algorithms like CLEAN-SC,
            where not the full transfer matrix is needed
        
        Returns
        -------
        array of complex128
            array of shape (ngridpts, nmics) containing the steering vectors for the given frequency
        """
        func = {'classic' : lambda x: x / absolute(x) / x.shape[-1],
                'inverse' : lambda x: 1. / x.conj() / x.shape[-1],
                'true level' : lambda x: x / einsum('ij,ij->i',x,x.conj())[:,newaxis],
                'true location' : lambda x: x / sqrt(einsum('ij,ij->i',x,x.conj()) * x.shape[-1])[:,newaxis]
                }[self.steer_type]
        return func(self.transfer(f, ind))

class SyntheticVerb(PointSourceIsm):
    
    def result(self, num=128):
        """
        Python generator that yields the output at microphones block-wise.
                
        Parameters
        ----------
        num : integer, defaults to 128
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block) .
        
        Returns
        -------
        Samples in blocks of shape (num, numchannels). 
            The last block may be shorter than num.
        """
        signal = self.signal.usignal(self.up)
        out = zeros((num, self.numchannels))
        h = self.impulse_response(self.loc)
        y = empty((self.numsamples,self.numchannels))
        for j in range(0,self.numchannels):
            y[:,j] = convolve(signal,h[:,j])
        i = 0
        ind = 0
        ts = self.start_t*self.sample_freq
        ts = rint(ts).astype(int)
        tm = self.start*self.sample_freq
        tm = rint(tm).astype(int)
        n = y.shape[0] + ts
        while n:
            n -= 1
            if ts>0:
                if tm>0:
                    tm-=1
                else:
                    out[i]=0.0
                    i+=1
                    if i == num:
                        yield out
                        i = 0
                ts-=1
            else:
                try:
                    out[i] = y[(ind+tm),:]
                    i += 1
                    ind += 1
                    if i == num:
                        yield out
                        i = 0
                except IndexError: #if no more samples available from the source
                    break
        if i > 0: # if there are still samples to yield
            yield out[:i]         

class SyntheticVerbMoving(MovingPointSourceIsm):
    def result(self, num=128):
        """
        Python generator that yields the output at microphones block-wise.
                
        Parameters
        ----------
        num : integer, defaults to 128
            This parameter defines the size of the blocks to be yielded
            (i.e. the number of samples per block).
        
        Returns
        -------
        Samples in blocks of shape (num, numchannels). 
            The last block may be shorter than num.
        """   
        signal = self.signal.usignal(self.up)
        out = zeros((num, self.numchannels))
        # shortcuts and intial values
        t = self.start*ones(self.mics.num_mics)
        i = 0
        ind = 0
        epslim = 0.1/self.up/self.sample_freq
        k = 0
        convsignaln = self.numsamples
        y = zeros((convsignaln,self.numchannels))
        ytemp = zeros((convsignaln,self.numchannels))

        #determine length n of result function (either shorter signal length or shorter trajectory length)
        numtrajectory = list(self.trajectory.points.items())[-1][0]
        if numtrajectory==self.numsamples/self.sample_freq:
            n = self.numsamples
        elif numtrajectory>=self.numsamples/self.sample_freq:
            n = self.numsamples
            temp = n/self.sample_freq
            warnstring = "time of signal (t=%f s) is shorter than time of trajectory (t=%f s)!" %(temp,numtrajectory)
            warnings.warn(warnstring)
        else:
            n = numtrajectory*self.sample_freq
            temp = self.numsamples/self.sample_freq
            warnstring = "time of trajectory (t=%f s) is shorter than time of signal (t=%f s)!" % (numtrajectory, temp)
            warnings.warn(warnstring)
        while n:
            h = self.impulse_response(epslim,t)
            ylen = self.numsamples+h.shape[0]
            ylen = rint(ylen).astype(int)
            if k<self.numsamples:
                if ylen>convsignaln:
                    n+=ylen-convsignaln
                    dim1 = (ylen-convsignaln)
                    ext = zeros((dim1,self.numchannels))
                    y = append(y,ext, 0)
                    convsignaln = ylen
                ytemp = zeros((convsignaln,self.numchannels))
                
                for j in range(0,self.numchannels):
                    c = convolve([signal[ind*self.up]],h[:,j])
                    ytemp[ind:ind+c.shape[0],j]=c
                
                y = y + ytemp

            t += 1./self.sample_freq
            try:
                out[i] = y[array(ind),:]
                i += 1
                ind += 1
                if i == num:
                    print("n: ",n)
                    yield out
                    i = 0
            except StopIteration:
                return
            k += 1
            n -= 1
        if i>0:
            return out[:j]
