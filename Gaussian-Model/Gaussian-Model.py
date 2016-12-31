#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Initialization 
file1 = '../UCI-Database/Skin.txt'
file2 = '../UCI-Database/NonSkin.txt'
TP = 0;
TN = 0;
FP = 0;
FN = 0;
sk = []
nonsk = []

# Function to convert from RGP to YCbCr
def RGB2YCrCb(r, g, b):
	Y = .299*r + .587*g + .114*b
	Cb = int(np.round(128 -.168736*r -.331364*g + .5*b))
	Cr = int(np.round(128 +.5*r - .418688*g - .081312*b))
	return (Cb, Cr)

#[C1, C2] = RGB2YCrCb(240, 180, 70)
#print(C1)
#print(C2)

with open(file1) as SkinFile:
	for line in SkinFile:
		[b, g, r, label] = line.split()
		B = int(b)
		G= int(g)
		R = int(r)
		[Cb, Cr] = RGB2YCrCb(R, G, B)
		sk.append([Cb, Cr])

ColorArray = np.array(sk) # convert the list "a" to numpy array "c"
[l, w] = ColorArray.shape
#print(l)

SkinTest  = ColorArray[0:int(l/2), : ] # Training data
SkinTrain = ColorArray[int(l/2): , : ]
print('Number of skin training data = %d' %(SkinTrain.shape[0]) )
print('Number of skin Testing data = %d' %(SkinTest.shape[0]) )

Cmean = SkinTrain.mean(axis = 0) # Calculate the mean along all Cb, Cr
print('Mean = ' + str(Cmean) )

x = SkinTrain - Cmean;
[M, N] = x.shape

CovMatrix = (1/M)*np.dot(x.T, x) 
print(CovMatrix)
#print(CovMatrix.shape)


x, y = np.mgrid[60:150:1, 135:180:1]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x; pos[:, :, 1] = y
rv = multivariate_normal(Cmean, CovMatrix)
allout = rv.pdf(pos)
plt.contourf(x, y, allout)


## Test 
threshold = 0.0001
for i in SkinTest:
	if (rv.pdf(i) > threshold):
		TP +=1;
	else:
		FN +=1;

print('TP = %d' %(TP))
print('TP Rate = %f %% ' %( (TP/SkinTest.shape[0])*100 )  )
print('FN Rate = %f %% ' %( (FN/SkinTest.shape[0])*100 )  )


with open(file2) as NonSkinFile:
	for line in NonSkinFile:
		[b, g, r, label] = line.split()
		B = int(b)
		G= int(g)
		R = int(r)
		[Cb, Cr] = RGB2YCrCb(R, G, B)
		nonsk.append([Cb, Cr])

NonSkinTest = np.array(nonsk) # convert the list "a" to numpy array "c"

for i in NonSkinTest:
	if (rv.pdf(i) > threshold):
		FP +=1;
	else:
		TN +=1;

print('FP = %d' %(FP))
print('FP Rate = %f %% ' %( (FP/NonSkinTest.shape[0])*100 )  )
print('TN Rate = %f %% ' %( (TN/NonSkinTest.shape[0])*100 )  )

plt.show()

# source of YCbCr conversion:
# https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion
# https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.multivariate_normal.html