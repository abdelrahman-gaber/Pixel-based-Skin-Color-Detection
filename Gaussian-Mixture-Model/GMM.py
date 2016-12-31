import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


from scipy.stats import multivariate_normal
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold

file1 = '../UCI-Database/Skin.txt'
file2 = '../UCI-Database/NonSkin.txt'

n_classes = 4;
cov_type = 'full'
sk = []
FullModel = []
Out =[]

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
		sk.append([Cb, Cr])

SkinColorArray = np.array(sk) # convert the list "sk" to numpy array "SkinColorArray"
[l, w] = SkinColorArray.shape
SkinTest = SkinColorArray[0:int(l/2), : ] # Training data
SkinTrain = SkinColorArray[int(l/2): , : ]
print('Number of skin training data = %d' %(SkinTrain.shape[0]) )
print('Number of skin Testing data = %d' %(SkinTest.shape[0]) )

Gmm = GaussianMixture(n_components=n_classes,
                   covariance_type=cov_type, max_iter=80,  n_init=10, random_state=0)

Gmm.fit(X=SkinTrain,y=None)

mean = Gmm.means_
cov = Gmm.covariances_
w = Gmm.weights_

print(mean)
print(cov)
print(w)
#FullModel = w[0]*multivariate_normal(mean[0], cov[0]) + w[1]*multivariate_normal(mean[1], cov[1]) + w[2]*multivariate_normal(mean[2], cov[2]) + w[3]*multivariate_normal(mean[3], cov[3])

# Remember: We need to use pdf() function in multivariate_normal to get a real data
for i in range(n_classes):
	FullModel.append(multivariate_normal(mean[i], cov[i]))

# Plot the Separate Gaussians with the estimated parameters 
x, y = np.mgrid[60:150:1, 135:180:1]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x; pos[:, :, 1] = y

print(x.shape)
print(y.shape)

GMM = np.zeros(x.shape)
for i in range(n_classes):
	GMM += w[i]*FullModel[i].pdf(pos)

for i in range(n_classes):
	Out.append(FullModel[i].pdf(pos))

for i in range(n_classes):
	plt.contour(x, y, Out[i])

plt.contourf(x, y, GMM)

plt.show()
