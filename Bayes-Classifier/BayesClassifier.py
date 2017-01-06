import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.naive_bayes import GaussianNB

# Initialization 
file1 = '../UCI-Database/Skin.txt'
file2 = '../UCI-Database/NonSkin.txt'
file3='../UCI-Database/Skin_NonSkin.txt'

Skin=[]
NonSkin=[]
SkinLabel=[]
NonSkinLabel=[]
Train=[]
Test=[]

with open(file1) as SkinFile:
	for line in SkinFile:
		[b1, g1, r1, l1] = line.split()
		B1 = int(b1)
		G1= int(g1)
		R1 = int(r1)
		label1=int(l1)
		Skin.append([B1,G1,R1])
		SkinLabel.append(label1)

with open(file2) as NonSkinFile:
	for line in NonSkinFile:
		[b2, g2, r2, l2] = line.split()
		B2 = int(b2)
		G2= int(g2)
		R2 = int(r2)
		label2=int(l2)
		NonSkin.append([B2,G2,R2])
		NonSkinLabel.append(label2)

SkinColor=np.array(Skin)
NonSkinColor=np.array(NonSkin)
[s,u]=SkinColor.shape
[ns,v]=NonSkinColor.shape

LabelSkin=np.array(SkinLabel)
LabelNonSkin=np.array(NonSkinLabel)

# Dividing Skin and Nonskin arrays to Test and Training sets
SkinTrain=SkinColor[0:int(s/2), : ]
LabelSkinTrain=LabelSkin[0:int(s/2)]
NonSkinTrain=NonSkinColor[0:int(ns/2), : ]
LabelNonSkinTrain=LabelNonSkin[0:int(ns/2)]

SkinTest=SkinColor[int(s/2): , : ]
LabelSkinTest=LabelSkin[int(s/2): ]
NonSkinTest=NonSkinColor[int(ns/2): , : ]
LabelNonSkinTest=LabelNonSkin[int(ns/2):]

# Train matrix = [SkinTrain,NonSkinTrain]
Train = np.concatenate((SkinTrain,NonSkinTrain),axis=0)
# Test matrix = [SkinTest,NonSkinTest]
Test = np.concatenate((SkinTest,NonSkinTest),axis=0)
# Train Label vector = [LabelSkinTrain,LabelNonSkinTrain]
LabelTrain=np.concatenate((LabelSkinTrain,LabelNonSkinTrain),axis=0)
# Test Label vector = [LabelSkinTest,LabelNonSkinTest]
LabelTest=np.concatenate((LabelSkinTest,LabelNonSkinTest),axis=0)

# The different naive Bayes classifiers differ mainly by the assumptions they 
# make regarding the distribution of P(x|y), Here we used the gaussian distribution
model = GaussianNB()

# Train the model using the training set 
model.fit(Train, LabelTrain)

# Predict Skin training data
predicted1 = model.predict(SkinTest)
# Predict NonSkin training data
predicted2 = model.predict(NonSkinTest)

TP = sum(predicted1 == LabelSkinTest)
TN =sum(predicted2 == LabelNonSkinTest)
FP = sum(predicted1 != LabelSkinTest)
FN =sum(predicted2 != LabelNonSkinTest)

TPR = float(TP)/(LabelSkinTest.shape[0]) * 100
TNR = float(TN)/(LabelNonSkinTest.shape[0]) * 100
FPR = float(FP)/(LabelSkinTest.shape[0]) * 100
FNR = float(FN)/(LabelNonSkinTest.shape[0]) * 100

print('TP Rate = ' + str(TPR))
print('FP Rate = ' + str(FPR))
#print('TN Rate = ' + str(TNR))
#print('FN Rate = ' + str(FNR))

# Reference:
# http://scikit-learn.org/stable/modules/naive_bayes.html