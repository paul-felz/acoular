"""
Verification Test 4 for Image Source Method

Generate a wall and a grid and check if the mirror grid is at right place.

@author: paul-felz
"""

from os import path

import unittest

from acoular import __file__ as bpath, config, MicGeom, WNoiseGenerator, PointSource, RectGrid3D, Room, GridExtender

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

gmirror = array([[2., 2., 1., 1., 2., 2., 1., 1.],
                 [2., 2., 2., 2., 1., 1., 1., 1.],
                 [0., 1., 0., 1., 0., 1., 0., 1.]])


class acoular_ism_test(unittest.TestCase):

    def test_mirror_grid(self):
        for i in range(gmirror.shape[0]):
            for j in range(gmirror.shape[1]):
                self.assertAlmostEqual(gmirror[i,j],gextend.mirrgrids[0][i,j],3)

if "__main__" == __name__:
    unittest.main() #exit=False
