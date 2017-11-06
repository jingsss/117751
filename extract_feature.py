#!/usr/bin/python
from pyAudioAnalysis import audioBasicIO
from pyAudioAnalysis import audioFeatureExtraction
import os
for subdir, dirs, files in os.walk("audio_segment/ice_age_a_mammoth_christmas"):
	for file in files:/Users/jing/Documents/Umich/2016winter/EECS498/HW1/temp.pl
		filepath = subdir + os.sep + file/Users/jing/Documents/Umich/2016winter/EECS498/HW1/temp.pl
		[Fs, x] = audioBasicIO.readAudioFile(filepath)
		F = audioFeatureExtraction.stFeatureExtraction(x, Fs, 0.050*Fs, 0.025*Fs)
		print F.shape