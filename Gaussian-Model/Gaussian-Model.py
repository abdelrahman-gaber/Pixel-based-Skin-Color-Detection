#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Initialization 
file1 = '../UCI-Database/Skin.txt'
file2 = '../UCI-Database/NonSkin.txt'


def RGB2YCrCb(r, g, b):
	Y = .299*r + .587*g + .114*b
	Cb = int(np.round(128 -.168736*r -.331364*g + .5*b))
	Cr = int(np.round(128 +.5*r - .418688*g - .081312*b))
	return (Cb, Cr)

#[C1, C2] = RGB2YCrCb(240, 180, 70)
#print(C1)
#print(C2)

a = []

with open(file1) as SkinFile:
	for line in SkinFile:
		[b, g, r, label] = line.split()
		B = int(b)
		G= int(g)
		R = int(r)
		[Cb, Cr] = RGB2YCrCb(R, G, B)
		a.append([Cb, Cr])

c = np.array(a) # convert the list "a" to numpy array "c"
print(c.shape)

Cmean = c.mean(axis = 0) # Calculate the mean along all Cb, Cr
print(Cmean)
print(Cmean.shape)

x = c - Cmean;
#print(x)
#print(x.shape)
[M, N] = x.shape

CovMatrix = (1/M)*np.dot(x.T, x) 
print(CovMatrix)
print(CovMatrix.shape)


x, y = np.mgrid[1:250:1, 1:250:1]
pos = np.empty(x.shape + (2,))
pos[:, :, 0] = x; pos[:, :, 1] = y
rv = multivariate_normal(Cmean, CovMatrix)
plt.contourf(x, y, rv.pdf(pos))
plt.show()

out = rv.pdf([100,150])
print(out)

#x = [150, 200]
#x = [np.linspace(50, 250, num = 100), np.linspace(50, 250, num = 100)]
#X = np.array(x).T
#y = multivariate_normal.pdf(x, mean= Cmean, cov= CovMatrix)
#plt.plot(X, y)
#plt.axis('equal')
#plt.show()
#print(y)


# source of YCbCr conversion:
# https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion
# https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.multivariate_normal.html