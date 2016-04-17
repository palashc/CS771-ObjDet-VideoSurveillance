import os
import cv2
import numpy as np
obj = ["Car","Person","Motorcycle","Bicycle","Rickshaw","Autorickshaw"]

def label(o):
	if o=="Person":
		ans = 0
	elif o=="Motorcycle":
		ans = 1
	elif o=="Bicycle":
		ans = 2
	elif o=="Car":
		ans = 3
	elif o=="Rickshaw":
		ans = 4
	elif o=="Autorickshaw":
		ans = 5
	return ans

y=[]
z = []
X = np.zeros((1,128))
sift = cv2.SIFT(100)



k = 0
for o in obj:
	path = "./data/" + o + "/"
	n = len(os.listdir(path))
	i=0
	for imagepath in os.listdir(path):
		print i,'/',n
		img = cv2.imread(path+imagepath)
		kp, des = sift.detectAndCompute(img,None)
		if des != None:
			X = np.concatenate((X,des),axis=0)
			for t in range(0,len(des)):
				y.append(label(o))
				z.append(k)
		k += 1
		i += 1


X.dump("X.pkl")
np.array(y).dump("y.pkl")
np.array(z).dump("z.pkl")
