#!/usr/bin/python
from scikits.audiolab import wavread
from pylab import *

signal, fs, enc = wavread('africa-toto.wav')
specgram(signal)
show()
