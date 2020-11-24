"""Verification Test 2 for Image Source Method (PointSourceIsm)

Test impulse Response Generator

@author: paul-felz
"""
from os import path

import unittest

from acoular import __file__ as bpath, config, MicGeom, Room, WNoiseGenerator, \
        Orientation, PointSourceIsm, FilterRIR

from numpy import argmax

config.global_caching = "none"

sfreq = 51200
duration = 1
nsamples = duration*sfreq

micgeofile = path.join(path.split(bpath)[0],'xml','tub_vogel64.xml')
mg = MicGeom( from_file=micgeofile )

cu = Room()
cu.create_wall_orientation(position=-0.88,orientation=Orientation.Y,alpha=0.0)

n1 = WNoiseGenerator( sample_freq=sfreq, numsamples=nsamples, seed=1 )
#HanningLPFilter
p1 = PointSourceIsm(filterrir=FilterRIR.HanningLpf, signal=n1, mics=mg, loc=(0,0.408,0.69), room=cu, up=1)
ir = p1.impulse_response((0,0.408,0.69))
ir63 = ir[:,63]

imp1 = argmax(ir63)
imp2 = argmax(ir63[imp1+1:])
imp2 = imp1+imp2

tdiff = (imp2-imp1)/sfreq

#RoundLPFilter
p2 = PointSourceIsm(filterrir=FilterRIR.RoundLpf, signal=n1, mics=mg, loc=(0,0.408,0.69), room=cu, up=1)
ir2 = p2.impulse_response((0,0.408,0.69))
ir263 = ir2[:,63]

imp1 = argmax(ir263)
imp2 = argmax(ir263[imp1+1:])
imp2 = imp1+imp2

tdiff2 = (imp2-imp1)/sfreq

class acoular_ism_impulse_response_test(unittest.TestCase):

    def test_impulse_response(self):
        self.assertAlmostEqual(tdiff,0.004,3)
        self.assertAlmostEqual(tdiff2,0.004,3)

if "__main__" == __name__:
    unittest.main() #exit=False
