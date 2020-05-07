"""Implements image source method of Jont B. Allen and David A. Berkley.

.. autosummary::
    :toctree: generated/
    
    Ism
"""

from numpy import array, zeros, concatenate, delete, where, floor, dot, subtract, \
pi, complex128, float32, sin, cos, isscalar, cross, sqrt
from numpy.linalg.linalg import norm

from traits.api import HasTraits, HasPrivateTraits, Float, Int, ListInt, ListFloat, \
CArray, Property, Instance, Trait, Bool, Range, Delegate, Enum, Any, \
cached_property, on_trait_change, property_depends_on, List, \
Tuple

from .internal import digest
from .sources import SamplesGenerator, SourceMixer, PointSource
from .grids import Grid, RectGrid
from .environments import Environment
from .microphones import MicGeom
from .fbeamform import SteeringVector

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
            desc="apsorption coefficient")

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

    point1 = Property()
    
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
    For example: A cuboid.
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


class Ism(SamplesGenerator):
    """
    Mirrors the source on walls of room and writes them to the list sources.
    J. Allen and D. Berkeley, "Image method for efficiently simulating small-room acoustics"
    with negative reflection coefficients adapted by
    E. Lehmann and A. Johansson, "Prediction of energy decay in room impulse responses simulated with an image-source model"
    """
    #TODO: Don't allow Mixer
    source = Instance(SamplesGenerator(),SamplesGenerator) 

    room = Trait(Room,
            desc="room with list of wall planes")

    #sources = Property(desc="mirrored sources")
    sources = Property()

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
    
    #TODO: Just allow extended Grids

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
 
    def transfer_room(self, f, ind=None):
        if ind is None:
            transm = 0.0
            for mirrgrid in self.grid.mirrgrids:
                transm = transm + calcTransfer(self.r0mirror, self.rmirror, array(2*pi*f/self.env.c))
            trans = calcTransfer(self.r0, self.rm, array(2*pi*f/self.env.c))
        elif not isinstance(ind,ndarray):
            trans = calcTransfer(self.r0[ind], self.rm[ind, :][newaxis], array(2*pi*f/self.env.c))#[0, :]
        else:
            trans = calcTransfer(self.r0[ind], self.rm[ind, :], array(2*pi*f/self.env.c))
        return trans

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
        elif not isinstance(ind,ndarray):
            trans = calcTransfer(self.r0[ind], self.rm[ind, :][newaxis], array(2*pi*f/self.env.c))#[0, :]
        else:
            trans = calcTransfer(self.r0[ind], self.rm[ind, :], array(2*pi*f/self.env.c))
        return trans
    
    def calcSteer_Formulation1AkaClassic_FullCSM(self, distGridToAllMics, waveNumber): 
        nMics = distGridToAllMics.shape[1]
        gridPointNum = distGridToAllMics.shape[0]
        steerVec = zeros((gridPointNum,nMics), complex128)
        for cntMics in range(nMics):
            expArg = float32(waveNumber * distGridToAllMics[:,cntMics])
            steerVec[:,cntMics] = (cos(expArg) - 1j * sin(expArg))
        return steerVec / (nMics)

    def calcIsmSteer_Formulation1AkaClassic(self, waveNumber): 
        nMics = self.rm.shape[1]
        gridPointNum = self.rm.shape[0]
        steerVec = zeros((gridPointNum,nMics), complex128)
        for cntMics in range(nMics):
            expArg = float32(waveNumber * (self.rm[:,cntMics]-self.r0))
            Argd =  (self.r0/self.rm[:,cntMics])*(cos(expArg) - 1j * sin(expArg))
            steerVec[:,cntMics] = Argd/abs(Argd)
        for mirrNum in range(len(self.grid.mirrgrids)):
            for cntMics in range(nMics):
                expArg = float32(waveNumber * (self.rmirror[mirrNum][:,cntMics]-self.r0mirror[mirrNum]))
                Argd = (self.r0mirror[mirrNum]/self.rmirror[mirrNum][:,cntMics])*(cos(expArg) - 1j * sin(expArg))
                steerVec[:,cntMics] += (1-self.room.walls[mirrNum].alpha) * Argd/abs(Argd)
        return steerVec / (nMics)

    #def calcIsmSteer_Formulation1AkaClassic(self, waveNumber): 
        

    def _get_steer_vector(self):
        def steer_vector(f): return self.calcIsmSteer_Formulation1AkaClassic(2*pi*f/self.env.c)
        return steer_vector

