# coding=UTF-8
#------------------------------------------------------------------------------
# Copyright (c) 2007-2019, Acoular Development Team.
#------------------------------------------------------------------------------

"""
The Acoular library: several classes for the implemetation of 
acoustic beamforming

A minimal usage example would be:

>>>    m = MicGeom(from_file='mic_geom.xml')
>>>    g = RectGrid(x_min=-0.8, x_max=-0.2, y_min=-0.1, y_max=0.3, z=0.8,\
 increment=0.01)
>>>    t1 = TimeSamples(name='measured_data.h5')
>>>   cal = Calib(from_file='calibration_data.xml')
>>>    f1 = EigSpectra(time_data=t1, block_size=256, window="Hanning",\
 overlap='75%', calib=cal)
>>>    e1 = BeamformerBase(freq_data=f1, grid=g, mpos=m, r_diag=False)
>>>    fr = 4000
>>>    L1 = L_p(e1.synthetic(fr, 0))

The classes in the module possess a number of automatic data update
capabilities. That is, only the traits must be set to get the results.
The calculation need not be triggered explicitely.

The classes are also GUI-aware, they know how to display a graphical user
interface. So by calling

>>>    object_name.configure_traits()

on object "object_name" the relevant traits of each instance object may
be edited graphically.

The traits could also be set explicitely in the program, either in the
constructor of an object:

>>>    m = MicGeom(from_file='mic_geom.xml')

or at a later time

>>>    m.from_file = 'another_mic_geom.xml'

where all objects that depend upon the specific trait will update their
output if necessary.
"""

from .version import __author__, __date__, __version__

import os 

# workaround for problems with pyqt5 support, may be removed in the future
try:
    import pyface.qt
except:
    os.environ['QT_API'] = 'pyqt' 

#     
from .configuration import config

from .fileimport import time_data_import, csv_import, td_import, \
bk_mat_import, datx_import
try:
    from .nidaqimport import nidaq_import
except:
    pass

from .h5cache import td_dir, cache_dir
from .tbeamform import IntegratorSectorTime, \
BeamformerTime, BeamformerTimeSq, BeamformerTimeTraj, BeamformerTimeSqTraj
from .tprocess import TimeInOut, MaskedTimeInOut, Mixer, TimeAverage, \
TimeReverse, TimePower, FiltFiltOctave, FiltOctave, TimeCache, WriteWAV, \
WriteH5, SpatialInterpolator, SpatialInterpolatorRotation, Trigger, \
SampleSplitter, AngleTracker, SpatialInterpolatorConstantRotation

from .calib import Calib
from .trajectory import Trajectory
from .grids import Grid, RectGrid, RectGrid3D
from .environments import cartToCyl, cylToCart, Environment, UniformFlowEnvironment, RotatingFlow, \
FlowField, OpenJet, SlotJet, GeneralFlowEnvironment
from .microphones import MicGeom
from .spectra import PowerSpectra, PowerSpectra as EigSpectra, synthetic
from .ism import Orientation, Wall, WallHesse, Wall3Points, WallOrientation, Room, \
Ism, SteeringVectorRoom, GridExtender, FilterRIR, PointSourceIsm, IsmRealImages, \
MovingPointSourceIsm, SyntheticVerb, SyntheticVerbMoving
from .mint import LoadSignal, MintRIR, MintRIRSimulation, MintRIRMeasurement, \
        MintRIRSimulationMoving, Mint, EvaluateMint
from .fbeamform import BeamformerBase, BeamformerCapon, BeamformerEig, \
BeamformerMusic, BeamformerDamas, BeamformerDamasPlus, BeamformerOrth,BeamformerCleansc, \
BeamformerCMF, BeamformerClean, BeamformerFunctional, BeamformerGIB, L_p, integrate, \
PointSpreadFunction, SteeringVector

from .sources import PointSource, MovingPointSource, SamplesGenerator, \
TimeSamples, MaskedTimeSamples, PointSourceDipole, UncorrelatedNoiseSource, \
SourceMixer
from .signals import SineGenerator, WNoiseGenerator, SignalGenerator,\
PNoiseGenerator
