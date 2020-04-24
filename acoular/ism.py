"""Implements image source method of Jont B. Allen and David A. Berkley.

.. autosummary::
    :toctree: generated/
    
    Ism
"""

from numpy import array, zeros, concatenate, delete, where, floor, dot, subtract, \
pi, complex128, float32, sin, cos

from traits.api import HasTraits, HasPrivateTraits, Float, Int, ListInt, ListFloat, \
CArray, Property, Instance, Trait, Bool, Range, Delegate, Enum, Any, \
cached_property, on_trait_change, property_depends_on, List

from .internal import digest
from .sources import SamplesGenerator, SourceMixer, PointSource
from .grids import Grid, RectGrid
from .microphones import MicGeom
from .environments import Environment, Orientation, Cuboid
from .fbeamform import SteeringVector

import pdb


class SteeringVectorRoom( SteeringVector ):

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
                    r0temp = self.env._r(mirrgrid._gpos)
                    r0i.append(r0temp)
                return r0i
        else:

            print("Expected GridExtender object!")
            raise ValueError

    def _get_rmirror ( self ):
        if isinstance(self.grid,GridExtender):
            rmi = []
            for mirrgrid in self.grid.mirrgrids:
                rmtemp = self.env._r(mirrgrid._gpos,self.mics.mpos)
                rmi.append(rmtemp)
            return rmi
        else:
            print("Expected GridExtender object!")
            raise ValueError
 
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
    
    def calcIsmSteer_Formulation1AkaClassic_FullCSM(self, distGridToAllMics, waveNumber): 
        nMics = distGridToAllMics.shape[1]
        gridPointNum = distGridToAllMics.shape[0]
        steerVec = zeros((gridPointNum,nMics), complex128)
        for cntMics in range(nMics):
            expArg = float32(waveNumber * distGridToAllMics[:,cntMics])
            steerVec[:,cntMics] = (cos(expArg) - 1j * sin(expArg))
        print(steerVec.shape) 
        return steerVec / (nMics)

    def _get_steer_vector(self):
        def steer_vector(f): return self.calcIsmSteer_Formulation1AkaClassic_FullCSM(self.rm,2*pi*f/self.env.c)
        return steer_vector

class IsmRoom(SamplesGenerator):
 
    #sources = Trait(List( Instance(SamplesGenerator, ()) ))

    #ERROR wenn als property
    #sources = Property(desc="mirrored sources")
    sources = List()

    #to be mirrored
    source = Instance(SamplesGenerator(),SamplesGenerator) 

    cuboid = Trait(Cuboid,
            desc="cuboid walls")

    #srcmix = Instance(SourceMixer(),SourceMixer)

    sample_freq = Delegate('source','sample_freq') 

    numchannels = Delegate('source','numchannels')

    numsamples = Delegate('source','numsamples')

    #ldigest = Property( depends_on = ['sources.digest', ])
    def __init__(self,source,cuboid):
        self.cuboid = cuboid
        self.source = source
        self._set_sources()
        HasTraits.__init__( self )

    digest = Property( 
            depends_on = ['source.digest', 'sources']
            )

    @cached_property
    def _get_digest( self ):
        return digest(self)

    def _get_sources(self):
        return self.sources

    def _set_sources(self):
        if self.source:     #todo: mixer
            self.sources = [self.source]
            for wall in self.cuboid.walls:
                temp = self.clone_traits()
                temp.source.loc = temp.mirror_loc(wall.n0,wall.point1)
                #temp.source.signal.seed += 1
                self.sources.extend([temp.source])
        return self.sources

                    
    def mirror_loc(self, n0, point1):
        d = dot(self.source.loc,n0) - dot(point1,n0)
        n02d = tuple(2*d*x for x in n0)
        mirror_loc = subtract(self.source.loc,n02d)
        mirror_loc = tuple(mirror_loc)
        return mirror_loc

    def result(self,num):
        gens = [i.result(num) for i in self.sources[1:]]
        for temp in self.sources[0].result(num):
            sh = temp.shape[0]
            for g in gens:
                temp1 = next(g)
                if temp.shape[0] > temp1.shape[0]:
                    temp = temp[:temp1.shape[0]]
                temp += temp1[:temp.shape[0]]
            yield temp
            if sh > temp.shape[0]:
                break


class GridExtender(Grid):
    
    grid = Instance(Grid(), Grid)
    
    cuboid = Trait(Cuboid,
            desc="cuboid walls")

    mirrgrids = List()

    def __init__(self,grid,cuboid):
        self.cuboid = cuboid
        self.grid = grid
        self._set_mirrgrids()
        HasTraits.__init__( self )

    digest = Property(
            depends_on = ['grid','_nxsteps','_nysteps']
            )

    @cached_property
    def _get_digest( self ):
        return digest( self )
    
    #def add_grid(self, grid):
    #    self.mirrgrids.append(grid) 

    def mirror_gridpoint(self, n0, basepoint, point):
        d = dot(point,n0) - dot(basepoint,n0)
        n02d = tuple(2*d*x for x in n0)
        mirr_gridpoint = subtract(point,n02d)
        mirr_gridpoint = tuple(mirr_gridpoint)
        return mirr_gridpoint

    def _get_gpos ( self ):
        return self.grid.gpos

    def _get_mirrgrids( self ):
        return self.mirrgrids()    

    def _set_mirrgrids( self ):
        gpos = self.grid.pos()
        for wall in self.cuboid.walls:
            temp = self.clone_traits()
            mirr_gpos = zeros([3,gpos.shape[1]])
            for i in range(0,mirr_gpos.shape[1]):
                gridpoint = (gpos[0,i],gpos[1,i],gpos[2,i])
                gridpoint_mirr = self.mirror_gridpoint(wall.n0, wall.point1, gridpoint)
                mirr_gpos[0,i] = gridpoint_mirr[0]
                mirr_gpos[1,i] = gridpoint_mirr[1]
                mirr_gpos[2,i] = gridpoint_mirr[2]
            temp.grid.gpos = mirr_gpos
            #TODO: new nxsteps size etc
            self.mirrgrids.extend([temp.grid])

    @property_depends_on('_nxsteps,_nysteps')
    def _get_size(self):
        #xwallnum = h
        #size = (xwallnum+1)*self.grid.pos().shape[1]
        return self.grid.size

    @property_depends_on('_nxsteps, _nysteps')
    def _get_shape ( self ):
        return self.grid.shape
    
    

    """
    def _set__nxsteps(self,xwallnum,gridmatchwall):
        self._nxsteps = (xwallnum+1)*self.grid.nxsteps-gridmatchwall

    def _set__nysteps(self,ywallnum,gridmatchwall):
        self._nysteps = (ywallnum+1)*self.grid.nysteps-gridmatchwall

    def extend (self) :
        x_min = self.gpos[0,:].min()
        x_max = self.gpos[0,:].max()
        y_min = self.gpos[1,:].min()
        y_max = self.gpos[1,:].max()
        return (x_min, x_max, y_min, y_max)
    """
"""
class Gridextender(Grid):
    
    grid = Instance(Grid(), Grid)

    _nxsteps = Int(1,
            desc="number of extended grid points along x-axis")
    
    _nysteps = Int(1,
            desc="number of extended grid points along y-axis")

    digest = Property(
            depends_on = ['grid','_nxsteps','_nysteps']
            )

    @cached_property
    def _get_digest( self ):
        return digest( self )
    
    @on_trait_change('grid')
    def _get_gpos( self ):
        walls = [-0.3,0.3]
        gpos = self.grid.pos()
        for j in range(0,len(walls)):
            mirr_temp = self.grid.pos()
            dim = int(floor(j/2))
            indices = where(mirr_temp[dim,:]==walls[j])[0]
            if len(indices)>0:
                a = delete(mirr_temp[0,:],indices)
                b = delete(mirr_temp[1,:],indices)
                c = delete(mirr_temp[2,:],indices)
                mirr_temp = array([a,b,c])
            if j <2:
                dim=0
                dim2=1
                dim3=2
                if len(indices)>0:
                    if j==0:
                        self._set__nxsteps(1,1)
                        self._set__nysteps(0,0)
                    elif j==1:
                        self._set__nxsteps(2,2)
                        self._set__nysteps(0,0)
                else:
                    if j==0:
                        self._set__nxsteps(0,1)
                        self._set__nysteps(0,0)
                    elif j==1:
                        self._set__nxsteps(0,2)
                        self._set__nysteps(0,0)
            elif j<4:
                dim=1
                dim2=0
                dim3=2
                if len(indices)>0:
                    self._set__nysteps(1)
                else:
                    self._set__nysteps(0)
            elif j<6:
                dim=2
                dim2=0
                dim3=1
            else:
                raise(ValueError("walls should be a list with maximum 6 entries"))
            mirr_gpos = zeros([3,mirr_temp.shape[1]])
            #if upper mirror
            for i in range(0,mirr_temp.shape[1]):
                cpind = mirr_temp.shape[1]-i-1
                #if grid outside of room
                #fail warn error
                #if it is a upper wall to grid
                if walls[j]>mirr_temp[dim,cpind]:    
                    mirr_gpos[dim,i] = walls[j] + abs(walls[j]-mirr_temp[0,cpind])
                #if it is a lower wall to grid
                elif walls[j]<mirr_temp[0,i]: 
                    mirr_gpos[dim,i] = walls[j] - abs(walls[j]-mirr_temp[0,cpind])    
            
            mirr_gpos[dim2,:]=mirr_temp[dim2,:] 
            mirr_gpos[dim3,:]=mirr_temp[dim3,:]
            if j%2:
                gpos = concatenate([gpos,mirr_gpos],1)
            else:
                gpos = concatenate([mirr_gpos,gpos],1)
        return gpos 

    @property_depends_on('_nxsteps,_nysteps')
    def _get_size(self):
        #xwallnum = 1
        #size = (xwallnum+1)*self.grid.pos().shape[1]
        return self._nxsteps*self._nysteps

    @property_depends_on('_nxsteps, _nysteps')
    def _get_shape ( self ):
        return (self._nxsteps, self._nysteps)

    def _set__nxsteps(self,xwallnum,gridmatchwall):
        self._nxsteps = (xwallnum+1)*self.grid.nxsteps-gridmatchwall

    def _set__nysteps(self,ywallnum,gridmatchwall):
        self._nysteps = (ywallnum+1)*self.grid.nysteps-gridmatchwall
    
    def extend (self) :
        x_min = self.gpos[0,:].min()
        x_max = self.gpos[0,:].max()
        y_min = self.gpos[1,:].min()
        y_max = self.gpos[1,:].max()
        return (x_min, x_max, y_min, y_max)
"""
