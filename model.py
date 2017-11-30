#!/usr/bin/python
from scipy.io import arff
import numpy as np

def read_from_arff(filename):
	with open(filename,'r') as f:
		data, meta = arff.loadarff(f)
		data = np.asarray([list(i) for i in data])
	return data, meta

data, meta =read_from_arff('Features/test.arff')
print data.shape

