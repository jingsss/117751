#!/usr/bin/python

from pydub import AudioSegment
import os,shutil
import numpy as np
def segment(label_filename, audio_filename, label_cnt, labels, directory):
	newAudio = AudioSegment.from_wav(audio_filename)
	newAudio = newAudio.set_channels(1)
	default = 0
	with open(label_filename) as f:
		f.readline()
		for l in f:
			line = l.strip().split(",")
			t1 = int(line[0])
			t2 = t1 + int(line[1])
			label = line[-1].strip().upper()
			if label not in labels:
				continue
			else:
				if label != 'N' or np.random.uniform(0,1) < 0.15:
					cnt = label_cnt.get(label, default)
					new_name = directory + "/" + label + "_" + str(cnt) + ".wav"
					label_cnt[label] = cnt + 1
					tmp = newAudio[t1:t2]
					tmp.export(new_name, format="wav")
	
def mkdir(directory):
	try:
		os.makedirs(directory)
	except:
		shutil.rmtree(directory)
		os.makedirs(directory)

		
audio_filename = ["audio/SHREK_2.wav", "audio/cartoons.wav","audio/ice_age_a_mammoth_christmas.wav", "audio/team_america.wav"]
filename = ['label/SHREK_2.csv','label/cartoons.csv','label/ice_age_a_mammoth_christmas.csv','label/team_america.csv']	
directory = "combine_data/train_resample"

#audio_filename = ["audio/chicken_run.wav", "audio/ice_age.wav"]
#filename = ['label/chicken_run.csv', 'label/ice_age.csv']	
#directory = "combine_data/test_resample"
mkdir(directory)

labels = ["N","J","S","F","A","C","D"]
label_cnt = dict()
default = 0

for i in range(len(audio_filename)):
	segment(filename[i], audio_filename[i], label_cnt, labels, directory)
	print label_cnt

