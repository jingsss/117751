#!/usr/bin/env python
from os import rename, listdir

badprefix = "cheese_"
dir1 = "tess/younger_talk_surprise/"
fnames = listdir(dir1)

for fname in fnames:
	print fname
	if fname.endswith(".wav"):
		rename(dir1+fname, dir1+"S_"+fname)