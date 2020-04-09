
"""Implements image source method of Jont B. Allen and David A. Berkley.

.. autosummary::
    :toctree: generated/
    
    Ism
"""

from numpy import array, zeros, concatenate, delete, where, floor, dot, subtract

from traits.api import HasTraits, HasPrivateTraits, Float, Int, ListInt, ListFloat, \
CArray, Property, Instance, Trait, Bool, Range, Delegate, Enum, Any, \
cached_property, on_trait_change, property_depends_on, List

from .internal import digest
from .sources import SamplesGenerator, SourceMixer, PointSource
from .grids import Grid, RectGrid
from .microphones import MicGeom
from .environments import Environment, Orientation, Cuboid

import pdb

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

class SteeringVectorRoom( HasPrivateTraits ):
    """ 
    Basic class for implementing steering vectors with monopole source transfer models
    """
    
    #: :class:`~acoular.grids.Grid`-derived object that provides the grid locations.
    grid = Trait(Grid, 
        desc="beamforming grid")
    
    #: :class:`~acoular.microphones.MicGeom` object that provides the microphone locations.
    mics = Trait(MicGeom, 
        desc="microphone geometry")

    #: Type of steering vectors, see also :ref:`Sarradj, 2012<Sarradj2012>`. Defaults to 'true level'.
    steer_type = Trait('true level', 'true location', 'classic', 'inverse',
                  desc="type of steering vectors used")
    
    #: :class:`~acoular.environments.Environment` or derived object, 
    #: which provides information about the sound propagation in the medium.
    #: Defaults to standard :class:`~acoular.environments.Environment` object.
    env = Instance(Environment(), Environment)
    
    # TODO: add caching capability for transfer function
    # Flag, if "True" (not default), the transfer function is 
    # cached in h5 files and does not have to be recomputed during subsequent 
    # program runs. 
    # Be aware that setting this to "True" may result in high memory usage.
    #cached = Bool(False, 
    #              desc="cache flag for transfer function")    
    
    
    # Sound travel distances from microphone array center to grid 
    # points or reference position (readonly). Feature may change.
    r0 = Property(desc="array center to grid distances")

    # Sound travel distances from array microphones to grid 
    # points (readonly). Feature may change.
    rm = Property(desc="all array mics to grid distances")
    
    # mirror trait for ref
    _ref = Any(array([0.,0.,0.]),
               desc="reference position or distance")
    
    #: Reference position or distance at which to evaluate the sound pressure 
    #: of a grid point. 
    #: If set to a scalar, this is used as reference distance to the grid points.
    #: If set to a vector, this is interpreted as x,y,z coordinates of the reference position.
    #: Defaults to [0.,0.,0.].
    ref = Property(desc="reference position or distance")
    
    def _set_ref (self, ref):
        if isscalar(ref):
            try:
                self._ref = absolute(float(ref))
            except:
                raise TraitError(args=self,
                                 name='ref', 
                                 info='Float or CArray(3,)',
                                 value=ref) 
        elif len(ref) == 3:
            self._ref = array(ref, dtype=float)
        else:
            raise TraitError(args=self,
                             name='ref', 
                             info='Float or CArray(3,)',
                             value=ref)
      
    def _get_ref (self):
        return self._ref
    
    
    # internal identifier
    digest = Property( 
        depends_on = ['steer_type', 'env.digest', 'grid.digest', 'mics.digest', '_ref'])
    
    # internal identifier, use for inverse methods, excluding steering vector type
    inv_digest = Property( 
        depends_on = ['env.digest', 'grid.digest', 'mics.digest', '_ref'])
        
    @property_depends_on('grid.digest, env.digest, _ref')
    def _get_r0 ( self ):
        if isscalar(self.ref) and self.ref > 0:
            return full((self.grid.size,), self.ref)
        else:
            return self.env._r(self.grid.pos())

    @property_depends_on('grid.digest, mics.digest, env.digest')
    def _get_rm ( self ):
        return self.env._r(self.grid.pos(), self.mics.mpos)
 
    @cached_property
    def _get_digest( self ):
        return digest( self )
    
    @cached_property
    def _get_inv_digest( self ):
        return digest( self )
    
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
        
        Returns
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

class GridExtender(Grid):
    
    grid = Instance(Grid(), Grid)
    
    cuboid = Trait(Cuboid,
            desc="cuboid walls")

    _nxsteps = Int(1,
            desc="number of extended grid points along x-axis")
    
    _nysteps = Int(1,
            desc="number of extended grid points along y-axis")

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
            pdb.set_trace()
            temp.grid.gpos = mirr_gpos
            #TODO: new nxsteps size etc
            self.mirrgrids.extend([temp.grid])

    @property_depends_on('_nxsteps,_nysteps')
    def _get_size(self):
        #xwallnum = h
        #size = (xwallnum+1)*self.grid.pos().shape[1]
        return self._nxsteps*self._nysteps

    @property_depends_on('_nxsteps, _nysteps')
    def _get_shape ( self ):
        return (self._nxsteps, self._nysteps)

    def _set__nxsteps(self):
        return self.grid.nxsteps

    def _set__nysteps(self):
        return self.grid.nysteps
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
