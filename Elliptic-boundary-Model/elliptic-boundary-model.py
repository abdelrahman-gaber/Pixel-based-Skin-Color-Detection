import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from numpy import dot
from numpy.linalg import inv
from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold

file = '../UCI-Database/Skin_NonSkin.txt'
file1 = '../UCI-Database/Skin.txt'
file2 = '../UCI-Database/NonSkin.txt'

sk1 = []
sk2 = []
TP = 0;
TN = 0;
FP = 0;
FN = 0;

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

SkinColorArray = np.array(sk1) # convert the list "sk1" to numpy array "SkinColorArray"
NonSkinColorArray = np.array(sk2) # convert the list "sk2" to numpy array "SkinColorArray"
AllColorArray = np.concatenate((SkinColorArray, NonSkinColorArray), axis=0)

[l, w] = SkinColorArray.shape
[N, M] = AllColorArray.shape
print('Number of elements in the dataset = %d' %(N) )

SkinTest  = SkinColorArray[0:int(l/2), : ] # Test data
SkinTrain = SkinColorArray[int(l/2): , : ] # Training data
print('Number of skin training data = %d' %(SkinTrain.shape[0]) )
print('Number of skin Testing data = %d' %(SkinTest.shape[0]) )
print('Number of Non skin Testing data = %d' %(NonSkinColorArray.shape[0]) )

# Calculating all parameters of the model 
# phi for the Skin training data only 
phi = (1/l)*SkinColorArray.sum(axis=0)
print('Phi = ' + str(phi))
#print(phi.shape)

# mue for all data (N)
mue = AllColorArray.mean(axis = 0)
print('mue = ' + str(mue))

# Calculate Lambdas
x = AllColorArray - mue;
[N, M] = x.shape
#print(N)
Lambda = (1/N)*np.dot(x.T, x) 
print('Lambda = \n' + str(Lambda))

LambdaInv = inv(Lambda)

#c = RGB2YCrCb(197,197,157)
#c = RGB2YCrCb(240, 180, 70)
#c = RGB2YCrCb(74,85,123)

Threshold = 1.0;

# Calculate the final elliptical model and test all skin and nonskin data
for c in SkinColorArray:
	z = c - phi
	zT = z.T 
	EllipticModel = zT.dot(LambdaInv).dot(z)
	#print(EllipticModel)
	if (EllipticModel < Threshold):
		TP += 1
	else:
		FN +=1

for c in NonSkinColorArray:
	z = c - phi
	zT = z.T 
	EllipticModel = zT.dot(LambdaInv).dot(z)
	#print(EllipticModel)
	if (EllipticModel > Threshold):
		FP += 1
	else:
		TN +=1


print('TP = %d' %(TP))
print('TP Rate = %f %% ' %( (TP/SkinColorArray.shape[0])*100 )  )
print('FN Rate = %f %% ' %( (FN/SkinColorArray.shape[0])*100 )  )

print('FP = %d' %(FP))
print('FP Rate = %f %% ' %( (FP/NonSkinColorArray.shape[0])*100 )  )
print('TN Rate = %f %% ' %( (TN/NonSkinColorArray.shape[0])*100 )  )


# Calculate the final elliptical model
#z = c - phi
#zT = z.T 
#EllipticModel = zT.dot(LambdaInv).dot(z)
#print(EllipticModel)




#x, y = np.mgrid[60:150:1, 135:180:1]
#print(x.shape)
#pos = np.empty(x.shape + (2,))
#pos[:, :, 0] = x; pos[:, :, 1] = y
#rv = multivariate_normal(Cmean, CovMatrix)
#allout = rv.pdf(pos)
#print(pos)
#print(pos.shape)

#print(LambdaInv.shape)
#print(zT.shape)
#print(z.shape)

#tmp = np.dot(LambdaInv, z)
#EllipticModel = y.T * tmp 

#plt.contourf(x, y, EllipticModel)
#plt.show()
'''
y = SkinColorArray - phi
tmp = np.dot(LambdaInv, y)
EllipticModel = y.T * tmp 

Cmean = SkinTrain.mean(axis = 0) # Calculate the mean along all Cb, Cr
print('Mean = ' + str(Cmean) )

x = SkinTrain - Cmean;
[M, N] = x.shape

CovMatrix = (1/M)*np.dot(x.T, x) 
print(CovMatrix)
#print(CovMatrix.shape)
'''
