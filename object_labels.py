from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.externals import joblib

X = np.load("X.pkl")
y = np.load("y.pkl")
z = np.load("z.pkl")

# vids = ['input_video_sample1', 'input_video_sample2', 'input_video_sample3']#, 'datasample1', 'nov92015-1', 'nov92015-2', 'dec21h1330']

km = joblib.load("SiftKMeans_mini_clf.pkl")
clusters = km.labels_

# y=np.zeros((1,))
# X = np.zeros((1,128))
#
# for vid in vids:
# 	X = np.concatenate((X,np.load('X'+vid+'.pkl')),axis=0)
# 	y = np.concatenate((y,np.load('y'+vid+'.pkl')))
#
# print X.shape
# print y.shape

n = len(np.unique(z))
print n
p = len(np.unique(clusters))
print p

K = np.zeros((1,p))
ky =[]
i=0
while i<len(z):
    k = np.zeros((1,p))
    j = z[i]
    while z[i] == j:
        k[0,clusters[i]]+=1
        i+=1

        if(i == len(z)):
            break

    K = np.concatenate((K,k),axis=0)
    ky.append(y[i-1])
    m,_ = K.shape
    print m,"/",n

print K.shape
print len(ky)

K = K[1:]

K.dump('features.pkl')
np.array(ky).dump('labels.pkl')
