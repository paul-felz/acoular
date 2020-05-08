"""Verification Test 1 for Image Source Method

Generate 3 Walls with different Classmethods and check their attributes.

"""

from os import path

import unittest

from acoular import Room, Orientation
from numpy import array

#Create Walls
room = Room()
room.create_wall_hesse(point1=(0.1,0.2,0.3),n0=(1.0,0.0,0.0),alpha=0.1).create_wall_3points(point1=(0.0,0.2,0.0),point2=(0.1,0.2,0.1),point3=(0.2,0.2,0.0),alpha=0.2).create_wall_orientation(position=0.3,orientation=Orientation.Z,alpha=0.3)

#Reference Values
alpha=[0.1, 0.2, 0.3]
point11=[0.1,0.2,0.3]
point12=[0.0,0.2,0.0]
point13=[0.0,0.0,0.3]

class acoular_wall_test(unittest.TestCase):

    def test_wall_number(self):
        self.assertAlmostEqual(len(room.walls)/3,1,3)

    def test_absorption(self):
        for i in range(0,len(room.walls)):
            self.assertAlmostEqual(room.walls[i].alpha/alpha[i],1,3)

    def test_basepoint(self):
        self.assertAlmostEqual(sum(room.walls[0].point1),sum(point11),3)
        self.assertAlmostEqual(sum(room.walls[1].point1),sum(point12),3)
        self.assertAlmostEqual(sum(room.walls[2].point1),sum(point13),3)

    def test_n0(self): 
        self.assertAlmostEqual(room.walls[0].n0[0],1,3)
        self.assertAlmostEqual(room.walls[1].n0[1],1,3)
        self.assertAlmostEqual(room.walls[2].n0[2],1,3)

if "__main__" == __name__:
    unittest.main() #exit=False

