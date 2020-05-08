"""Verification Test 2 for Image Source Method

Generate walls and check for mirror sources at right places.

"""

from os import path

import unittest

from acoular import __file__ as bpath, config, MicGeom, WNoiseGenerator, PointSource, Room, Ism

from numpy import sqrt
config.global_caching = "none"

sfreq = 51200
duration = 1
nsamples = duration*sfreq

micgeofile = path.join(path.split(bpath)[0],'xml','unit_test_1mic.xml')
mg = MicGeom( from_file=micgeofile )

n1 = WNoiseGenerator( sample_freq=sfreq, numsamples=nsamples, seed=1 )
p1 = PointSource( signal=n1, mics=mg, loc=(0.3,0.3,0.3) )

#Create Wall
onewall= Room()
onewall.create_wall_hesse(point1=(1.0,1.0,1.0),n0=(1/sqrt(3),1/sqrt(3),1/sqrt(3)),alpha=0.0)

#Create mirror source behind wall
ism = Ism(source=p1,room=onewall)

#Reference Values
original = [0.3,0.3,0.3]
reflect = [1.7,1.7,1.7]

class acoular_ism_test(unittest.TestCase):

    def test_first_reflection(self):
        self.assertAlmostEqual(sum(ism.sources[0].loc),sum(original),3)
        self.assertAlmostEqual(sum(ism.sources[1].loc),sum(reflect),3)

    #TODO: TEST result()
    #TODO: second order reflections

if "__main__" == __name__:
    unittest.main() #exit=False
