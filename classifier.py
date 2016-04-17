from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
from sklearn.externals import joblib

X = np.load('features.pkl')
y = np.load('labels.pkl')

print X.shape
print y.shape

clf = SVC(kernel='poly',degree=10)
# clf = AdaBoostClassifier(n_estimators=500)
clf.fit(X,y)
print clf.score(X,y)
joblib.dump(clf, "SVC_p10.pkl", compress=3)

# clf = RandomForestClassifier(n_jobs=8)
# clf.fit(X,y)
# print clf.score(X,y)
# joblib.dump(clf, "RFC_clf.pkl", compress=3)
