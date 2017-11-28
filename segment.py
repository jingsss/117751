#!/usr/bin/python
from pydub import AudioSegment
import os
def mkdir(directory):
	try:
		os.makedirs(directory)
	except:
		pass
		
audio_filename = "audio/SHREK_2.wav"
filename = 'label/SHREK_2.csv'		
pre_directory = "audio_segment"
directory = pre_directory + "/" + audio_filename.split("/")[-1].split(".")[0].lower()
mkdir(directory)

labels = ["N","J","S","F","A","C","D"]
newAudio = AudioSegment.from_wav(audio_filename)
newAudio = newAudio.set_channels(1)
label_cnt = dict()
default = 0
with open(filename) as f:
	f.readline()
	for l in f:
		line = l.strip().split(",")
		t1 = int(line[0])
		t2 = t1 + int(line[1])
		label = line[-1].strip().upper()
		if label not in labels:
#			print label
			continue
		else:
			cnt = label_cnt.get(label, default)
			new_name = directory + "/" + label + "_" + str(cnt) + ".wav"
			label_cnt[label] = cnt + 1
			tmp = newAudio[t1:t2]
			tmp.export(new_name, format="wav")






			