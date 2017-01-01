import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold

file = '../UCI-Database/Skin_NonSkin.txt'
file1 = '../UCI-Database/Skin.txt'
file2 = '../UCI-Database/NonSkin.txt'

sk1 = []
sk2 = []

# Function to convert from RGP to YCbCr
def RGB2YCrCb(r, g, b):
	Y = .299*r + .587*g + .114*b
	Cb = int(np.round(128 -.168736*r -.331364*g + .5*b))
	Cr = int(np.round(128 +.5*r - .418688*g - .081312*b))
	return (Cb, Cr)

# Read Skin points and convert them to YCrCb
with open(file1) as SkinFile:
	for line in SkinFile:
		[b, g, r, label] = line.split()
		B = int(b)
		G= int(g)
		R = int(r)
		[Cb, Cr] = RGB2YCrCb(R, G, B)
		sk1.append([Cb, Cr])

# Read Non-skin points and convert them to YCrCb
with open(file2) as NonSkinFile:
	for line in NonSkinFile:
		[b, g, r, label] = line.split()
		B = int(b)
		G= int(g)
		R = int(r)
		[Cb, Cr] = RGB2YCrCb(R, G, B)
		sk2.append([Cb, Cr])

SkinColorArray = np.array(sk1) # convert the list "sk" to numpy array "SkinColorArray"
NonSkinColorArray = np.array(sk2) # convert the list "sk" to numpy array "SkinColorArray"
AllColorArray = np.concatenate((SkinColorArray, NonSkinColorArray), axis=0)

[l, w] = SkinColorArray.shape
[N, M] = AllColorArray.shape
print('Number of all training data = %d' %(N) )

SkinTest  = SkinColorArray[0:int(l/2), : ] 
SkinTrain = SkinColorArray[int(l/2): , : ] # Training data
print('Number of skin training data = %d' %(SkinTrain.shape[0]) )
print('Number of skin Testing data = %d' %(SkinTest.shape[0]) )
print('Number of Non skin Testing data = %d' %(NonSkinColorArray.shape[0]) )

# phi for the Skin training data only
phi = (1/l)*SkinColorArray.sum(axis=0)
print(phi)
print(phi.shape)

# mue for all data (N)
mue = AllColorArray.mean(axis = 0)
print(mue)

# 
x = AllColorArray - mue;
[N, M] = x.shape
print(N)
Lambda = (1/N)*np.dot(x.T, x) 
print(Lambda)

LambdaInv = np.inv(Lambda)



x, y = np.mgrid[60:150:1, 135:180:1]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x; pos[:, :, 1] = y
rv = multivariate_normal(Cmean, CovMatrix)
allout = rv.pdf(pos)
plt.contourf(x, y, allout)


y = SkinColorArray - phi
tmp = np.dot(LambdaInv, y)
EllipticModel = y.T * tmp 

'''
Cmean = SkinTrain.mean(axis = 0) # Calculate the mean along all Cb, Cr
print('Mean = ' + str(Cmean) )

x = SkinTrain - Cmean;
[M, N] = x.shape

CovMatrix = (1/M)*np.dot(x.T, x) 
print(CovMatrix)
#print(CovMatrix.shape)
'''
