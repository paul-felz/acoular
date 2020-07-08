"""Verification Test 5 for Image Source Method

@author: paul-felz
"""
from os import path

import unittest

from acoular import __file__ as bpath, config, MicGeom, Orientation, WNoiseGenerator, PointSource, SourceMixer, Room, PointSourceIsm, MaskedTimeInOut, PowerSpectra

config.global_caching = "none"

sfreq = 51200
duration = 1
nsamples = duration*sfreq


#micgeofile = path.join(path.split(bpath)[0],'xml','array64.xml')
micgeofile = path.join(path.split(bpath)[0],'xml','unit_test_1mic.xml')
mg = MicGeom( from_file=micgeofile )

n1 = WNoiseGenerator( sample_freq=sfreq, numsamples=nsamples, seed=1 )
p1 = PointSource( signal=n1, mics=mg, loc=(0.2,0.0,0.0) )
#p2 = PointSource( signal=n1, mics=mg, loc=(0.4,0.0,0.0) )
#pa = SourceMixer( sources=[p1, p2])

#Create Wall
onewall= Room()
onewall.create_wall_orientation(position=0.3,orientation=Orientation.X,alpha=0.0)

#Create mirror source behind wall
#ism = IsmRealImages(source=p1,room=onewall)
ism = PointSourceIsm(signal=n1, mics=mg, loc=(0.2,0,0.0), room=onewall, up=16)

ts1 = MaskedTimeInOut(source=p1)
ps1 = PowerSpectra( time_data=ts1, block_size=128, window='Hanning' ) 

ts2 = MaskedTimeInOut(source=ism)
ps2 = PowerSpectra( time_data=ts2, block_size=128, window='Hanning' ) 

class acoular_ism_result_test(unittest.TestCase):

    def test_result_csm(self):
        for i in range(ps1.csm.shape[0]):
            self.assertAlmostEqual(abs(ps1.csm[i,0,0]),abs(ps2.csm[i,0,0]),2)


if "__main__" == __name__:
    unittest.main() #exit=False
