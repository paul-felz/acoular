"""
Verification Test for SteeringVectorRoom

Generate a wall and a grid and check r0, rm and r0mirror, rmirror of SteeringVector vs. SteeringVectorRoom.

@author: paul-felz
"""

from os import path

import unittest

from acoular import __file__ as bpath, config, MicGeom, WNoiseGenerator, PointSource, RectGrid3D, Room, GridExtender, SteeringVector, SteeringVectorRoom

from numpy import array, sqrt
config.global_caching = "none"

sfreq = 51200
duration = 1
nsamples = duration*sfreq

micgeofile = path.join(path.split(bpath)[0],'xml','unit_test_1mic.xml')
mg = MicGeom( from_file=micgeofile )

n1 = WNoiseGenerator( sample_freq=sfreq, numsamples=nsamples, seed=1 )
p1 = PointSource( signal=n1, mics=mg, loc=(0.3,0.3,0.3) )

rg = RectGrid3D(x_min=0., x_max=1., y_min=0., y_max=1., z_min=0., z_max=1., increment=1.) 

#Create Wall
onewall= Room()
onewall.create_wall_hesse(point1=(1.0,1.0,1.0),n0=(1/sqrt(2),1/sqrt(2),0),alpha=0.0)

gextend = GridExtender(grid=rg,room=onewall) 

r0 = [0.,1.,1.,sqrt(2),1.,sqrt(2),sqrt(2),sqrt(3)]
r0mirror = [sqrt(8),3.,sqrt(5),sqrt(6),sqrt(5),sqrt(6),sqrt(2),sqrt(3)]

st = SteeringVector(grid=rg,mics=mg,steer_type='classic')
stro = SteeringVectorRoom(grid=gextend,mics=mg,room=onewall,steer_type='classic')

class acoular_ism_test(unittest.TestCase):

    def test_mirror_grid(self):
        self.assertEqual(len(st.r0),8)
        self.assertEqual(len(stro.r0mirror[0]),8)
        for i in range(0,len(r0)):
            self.assertAlmostEqual(r0[i],st.r0[i],3)
            self.assertAlmostEqual(r0mirror[i],stro.r0mirror[0][i],3)

if "__main__" == __name__:
    unittest.main() #exit=False
