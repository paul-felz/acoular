
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
from .environments import Orientation, Cuboid


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
        if self.source:
            self.sources = [self.source]
            for wall in self.cuboid.walls:
                temp = self.clone_traits()
                temp.source.loc = temp.mirror_loc(wall.n0,wall.point1)
                #temp.source.signal.seed += 1
                self.sources.extend([temp.source])
                print(self.sources)
        return self.sources

                    
    """
    def _set_sources( self ):
        if self.source:
            self.sources = [self.source]
            for wall in self.cuboid.walls:
                #position = wall.get_position()
                #orientation = wall.get_orientation()
                temp = self.clone_traits()
                temp.source.loc = temp.mirror_loc(wall.position,wall.orientation)
                temp.source.signal.seed += 1
                self.sources.extend([temp.source])
                print(self.sources)
        return self.sources
    """

    """
    def mirror_direction(self,loc,position,orientation):        
        #if upper mirror wall
        if position>loc[orientation]:
            loc[orientation]= position + abs(position-loc[orientation])
        #if lower mirror wall
        elif position<loc[orientation]:
            loc[orientation]= position - abs(position-loc[orientation])
        print(loc)
        return loc
    """

    def mirror_loc(self, n0, point1):
        d = dot(self.source.loc,n0) - dot(point1,n0)
        print(d)
        print(n0)
        n02d = tuple(2*d*x for x in n0)
        mirror_loc = subtract(self.source.loc,n02d)
        mirror_loc = tuple(mirror_loc)
        print(mirror_loc)
        return mirror_loc

    """
    def mirror_loc(self, position, orientation):
        mirror_loc = self.source.loc
        mirror_loc = array(mirror_loc)
        if orientation == Orientation.X:
            mirror_loc = self.mirror_direction(mirror_loc,position,0)
        elif orientation == Orientation.Y:
            mirror_loc = self.mirror_direction(mirror_loc,position,1)
        else:
            mirror_loc = self.mirror_direction(mirror_loc,position,2)
        mirror_loc = tuple(mirror_loc)
        return mirror_loc
    """

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

