#!/usr/bin/python

from pydub import AudioSegment
import os,shutil
from os import listdir
from os.path import isfile, join

AUDIO_SOURCE_FOLDER = "audio/"
SEGMENT_FOLDER = "audio_segments/"
LABEL_FILE_FOLDER = "label/"



#audio_filename = ["audio/SHREK_2.wav", "audio/cartoons.wav","audio/ice_age.wav"]
#filename = ['label/SHREK_2.csv','label/cartoons.csv','label/ice_age.csv']	
#directory = "combine_data/train"

#audio_filename = ["audio/ice_age.wav"]
#filename = ['label/ice_age.csv']	
#directory = "combine_data/test"

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

def speech_modification(filename, speed_rate, new_file_path):
	 cmdstring = "sox %s %s tempo %f" % (filename, new_file_path, speed_rate)
	 print cmdstring
	 os.system(cmdstring)


def all_files(audio_folder):
	audio_files = [f for f in listdir(audio_folder) if isfile(join(audio_folder, f)) and f.endswith("wav")]
	label_files = []
	for audio_file in audio_files:
		label_files.append(LABEL_FILE_FOLDER+audio_file.split(".wav")[0]+".csv")
	audio_files = [AUDIO_SOURCE_FOLDER + audio_file for audio_file in audio_files]

	return audio_files, label_files

def extract_original_segments(audio_files, label_files, label_cnt, labels):
	for i in range(len(audio_files)):
		audio_file = audio_files[i]
		label_file = label_files[i]
		# rates suggested by http://speak.clsp.jhu.edu/uploads/publications/papers/1050_pdf.pdf
		faster_rate = 1.1
		slower_rare = 0.9
		dir_orig = SEGMENT_FOLDER + audio_file.split(".wav")[0].split("/")[1] + "/Original/"
		dir_faster =  SEGMENT_FOLDER + audio_file.split(".wav")[0].split("/")[1] + "/faster/"
		dir_slower =  SEGMENT_FOLDER + audio_file.split(".wav")[0].split("/")[1] + "/slower/"
		mkdir(dir_orig)
		mkdir(dir_faster)
		mkdir(dir_slower)
		segment(label_file, audio_file, label_cnt, labels, directory=dir_orig)
		original_files = [f for f in listdir(dir_orig) if isfile(join(dir_orig, f)) and f.endswith("wav")]
		for original_file in original_files:
			speech_modification(dir_orig+original_file, faster_rate, dir_faster+original_file)
			speech_modification(dir_orig+original_file, slower_rare, dir_slower+original_file)
	


if __name__ == '__main__':
	#mkdir(directory)
	if False:
		labels = ["N","J","S","F","A","C","D"]
		label_cnt = dict()
		default = 0
		for i in range(len(audio_filename)):
			segment(filename[i], audio_filename[i], label_cnt, labels, directory)
			print label_cnt
	labels = ["N","J","S","F","A","C","D"]
	label_cnt = dict()
	audio_files, label_files = all_files(AUDIO_SOURCE_FOLDER)
	extract_original_segments(audio_files, label_files, label_cnt, labels)
