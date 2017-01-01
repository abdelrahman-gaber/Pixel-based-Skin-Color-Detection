import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold

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


with open(file1) as SkinFile:
	for line in SkinFile:
		[b, g, r, label] = line.split()
		B = int(b)
		G= int(g)
		R = int(r)
		[Cb, Cr] = RGB2YCrCb(R, G, B)
		sk1.append([Cb, Cr])


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

plt.scatter(NonSkinColorArray[:,0], NonSkinColorArray[:,1],marker='o', color='red')
plt.scatter(SkinColorArray[:,0], SkinColorArray[:,1],marker='x', color='blue')
#plt.plot(SkinColorArray[:,0], SkinColorArray[:,1])
plt.show()







