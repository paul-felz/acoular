"""Verification Test 1 for PointSourceIsm

@author: paul-felz
"""
from os import path

import unittest

from acoular import __file__ as bpath, config, MicGeom, Orientation, WNoiseGenerator, PointSource, SourceMixer, Room, PointSourceIsm, MaskedTimeInOut, PowerSpectra, \
        RectGrid, SteeringVector, BeamformerBase, L_p, IsmRealImages

config.global_caching = "none"

sfreq = 51200
duration = 1
nsamples = duration*sfreq


#micgeofile = path.join(path.split(bpath)[0],'xml','array64.xml')
micgeofile = path.join(path.split(bpath)[0],'xml','tub_vogel64.xml')
mg = MicGeom( from_file=micgeofile )

#room grid
rg = RectGrid(x_min=-0.3,x_max=0.3,y_min=-0.3,y_max=0.3,z=0.3,increment=0.02)

#Source
n1 = WNoiseGenerator( sample_freq=sfreq, numsamples=nsamples, seed=1 )
p1 = PointSource( signal=n1, mics=mg, loc=(0.0,0.3,0.3) )

#Create Wall
onewall= Room()
onewall.create_wall_orientation(position=0.3,orientation=Orientation.Y,alpha=0.0)

#Create mirror source behind wall
ism = PointSourceIsm(signal=n1, mics=mg, loc=(0.0,0.3,0.3), room=onewall, up=1)

#steering vector
st = SteeringVector(grid=rg, mics=mg, \
        steer_type='classic')

ts1 = MaskedTimeInOut(source=p1)
ps1 = PowerSpectra( time_data=ts1, block_size=128, window='Hanning' ) 
bb = BeamformerBase( freq_data=ps1, steer=st )
pm = bb.synthetic( 8000, 3 )
Lm1 = L_p( pm )

ts2 = MaskedTimeInOut(source=ism)
ps2 = PowerSpectra( time_data=ts2, block_size=128, window='Hanning' ) 
bb = BeamformerBase( freq_data=ps2, steer=st )
pm = bb.synthetic( 8000, 3 )
Lm2 = L_p( pm )

differencedB = Lm2.max()-Lm1.max()

class acoular_ism_result_test(unittest.TestCase):

    def test_ism_result(self):
        self.assertTrue(5 <= differencedB <= 7)


if "__main__" == __name__:
    unittest.main() #exit=False
