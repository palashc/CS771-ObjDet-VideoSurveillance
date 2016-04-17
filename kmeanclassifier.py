from sklearn.svm import LinearSVC
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.externals import joblib



X = np.load("X.pkl")
print X.shape
# print y.shape

n = 500

km = MiniBatchKMeans(n_clusters=n)
clabels = km.fit_predict(X)

joblib.dump(km, "SiftKMeans_mini_clf.pkl", compress=3)
