#!/usr/bin/env python
import numpy as np

# Initialization 
file1 = '../UCI-Database/Skin.txt'
file2 = '../UCI-Database/NonSkin.txt'
Scount = 0
Ncount = 0
TP = 0
FP = 0

with open(file1) as SkinFile:
	for line in SkinFile:
		[b, g, r, label] = line.split()
		B = int(b)
		G= int(g)
		R = int(r)
		Scount += 1

		if (R>95 and G>40 and B>20) and (max(R,G,B)-min(R,G,B)>15) and (abs(R-G)>15 and R>G and R>B):
			TP += 1;

		#print(line)
		#print ('count = %d' %(count) )

with open(file2) as NonSkinFile:
	for line in NonSkinFile:
		[b, g, r, label] = line.split()
		B = int(b)
		G= int(g)
		R = int(r)
		Ncount += 1
		
		if (R>95 and G>40 and B>20) and (max(R,G,B)-min(R,G,B)>15) and (abs(R-G)>15 and R>G and R>B):
			FP += 1;

#print(label)
#print(TP)
print ('Actual skin pixels = %d' %(Scount) )
print ('Actual non skin Pixels = %d' %(Ncount) )

print ('TP = %d' %(TP) )
TPrate = (TP/Scount)*100 
print ('TP Rate = %f %%' %(TPrate) )

print ('FP = %d' %(FP) )
FPrate = (FP/Ncount)*100
print ('FP Rate = %f %% ' % (FPrate) )