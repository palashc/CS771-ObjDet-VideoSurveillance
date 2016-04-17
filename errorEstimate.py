import json
import os
import cv2
import numpy as np
from sklearn.externals import joblib
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier

trainingData = {} #trainingData[video_name][frameNo] will have the objects in each frameNo of video_name
dump = ['dump/input_video_sample1.json', 'dump/input_video_sample2.json', 'dump/datasample1.json']#, 'dump/nov92015-1.json', 'dump/nov92015-2.json', 'dump/dec21h1330.json']
# dump = ['dump/datasample1.json']#, 'dump/nov92015-1.json', 'dump/nov92015-2.json', 'dump/dec21h1330.json']
for f in dump:
	# name = f[5:]
	# name = name[:-5]
	name = f
	trainingData[name] = {}
	print f
	with open(f) as json_data:
	        d = json.load(json_data)
	        for item in d:
	            label = d[item]["label"]
	            for frame in d[item]["boxes"]:
	                if not int(frame) in trainingData[name]:
	                    trainingData[name][int(frame)] = []
	                #print frame, label
	                temp_list = [d[item]["boxes"][frame]["xtl"], d[item]["boxes"][frame]["ytl"], d[item]["boxes"][frame]["xbr"], d[item]["boxes"][frame]["ybr"], label]
	                trainingData[name][int(frame)].append(temp_list)

# frame:[xlow ylow xhigh yhigh label]

# # #0, 232, 513, 721
# image = "./videoData2/input_video_sample2/%04d.jpg" % 731
# img = cv2.imread(image)
# crop_img = img[232:721, 0:513] # Crop from x, y, w, h -> 100, 200, 300, 400
# # NOTE: its img[y: y + h, x: x + w] and *not* img[x: x + w, y: y + h]
# cv2.imshow("cropped", crop_img)
# cv2.waitKey(0)

# sift = cv2.xfeatures2d.SIFT_create(150)
y=[]
X = np.zeros((1,128))
def label(o):           #I have no idea what labels to give
	if o=="Person":
		ans = 0
	elif o=="Motorcycle":
		ans = 1
	elif o=="Bicycle":
		ans = 2
	# elif o=="Number-plate":
	# 	ans = 3
	elif o=="Car":
		ans = 3
	elif o=="Rickshaw":
		ans = 4
	elif o=="Autorickshaw":
		ans = 5
	return ans

# print trainingData['dump/datasample1.json']

algo_list=['ada100','ada500','rfc','svc_rbf','svc_sigmoid']
# algo_list=['bgfg']
for algo in algo_list:
	pkl = ['pkl/new/input_video_sample1.mov_'+str(algo)+'.pkl', 'pkl/new/input_video_sample2.mov_'+str(algo)+'.pkl', 'pkl/new/datasample1.mov_'+str(algo)+'.pkl']
	# pkl = ['pkl/new/datasample1.mov_'+str(algo)+'.pkl']
	testingData = {}

	for i in pkl:
		testingData[i] = joblib.load(i).astype(int) # f x y w h l

	count = [0.0,0.0]
	Count=[0.0,0.0]
	ratioCount=[0.0,0.0]
	personCount=[0.0,0.0,0.0,0.0]
	motorcycleCount=[0.0,0.0,0.0,0.0]
	bicycleCount=[0.0,0.0,0.0,0.0]
	carCount=[0.0,0.0,0.0,0.0]
	rickshawCount=[0.0,0.0,0.0,0.0]
	autorickshawCount=[0.0,0.0,0.0,0.0]
	twowheelerCount=[0.0,0.0,0.0,0.0]
	threewheelerCount=[0.0,0.0,0.0,0.0]
	confusionMatrix=[[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0],[0,0,0,0,0,0]]
	confusionMatrix2=[[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]
	confusionMatrix3=[[0,0,0],[0,0,0],[0,0,0]]

	for i,j in zip(dump,pkl):
		for test in testingData[j]:
			if test[0] not in trainingData[i]:
				# count[0]+=1
				kk=0
			else:
				# count[1]+=1
				x = [test[1],test[1]+test[3]]
				y = [test[2],test[2]+test[4]]
				# x = [2*test[1],2*test[1]+2*test[3]]
				# y = [2*test[2],2*test[2]+2*test[4]]
				if x[1]-x[0] > 750 or y[1]-y[0] > 750:
					continue
				maxratio = 0.0
				maxoverlap = 0
				maxunion = 0
				mindist = 999999999
				minbox = None
				for box in trainingData[i][test[0]]:
					if str(box[4])=="Number-plate":
						continue
					# if test[5]==label(str(box[4])):
					x_overlap = max(0, min(x[1],box[2]) - max(x[0],box[0]))
					y_overlap = max(0, min(y[1],box[3]) - max(y[0],box[1]))
					overlapArea = x_overlap * y_overlap*1.0
					# unionArea = ((box[2]-box[0])*(box[3]-box[1]))+((x[1]-x[0])*(y[1]-y[0]))-overlapArea #Detection (Overlap) 0.248809654779 #Detection (Ratio) 0.272687952186
					unionArea = (((box[2]-box[0])*(box[3]-box[1]))+((x[1]-x[0])*(y[1]-y[0])))/2 #Detection (Overlap) 0.398475567681 #Detection (Ratio) 0.389941805599
					# unionArea = min(((box[2]-box[0])*(box[3]-box[1])),((x[1]-x[0])*(y[1]-y[0]))) #Detection (Overlap) 0.689018634647 Detection (Ratio) 0.526973891161
					ratio = overlapArea/unionArea
					dist = pow(box[0]-x[0],2)+pow(box[1]-y[0],2)+pow(box[2]-x[1],2)+pow(box[3]-y[1],2)
					if ratio > maxratio:
					# if mindist > dist: # Not Good
						maxoverlap = overlapArea
						maxunion = unionArea
						maxratio = ratio
						mindist = dist
						minbox = box
				# print test[0],'\t\t\t\t',x[0],y[0],x[1],y[1],test[5],'\t\t\t\t',minbox
				count[0]+= maxoverlap
				count[1]+= maxunion
				
				ratioCount[1]+=1
				if maxratio>0.5:
					ratioCount[0]+=1
				elif maxratio>0.25:
					ratioCount[0]+=0
				# else:
				# 	ratioCount[0]+=0

				if minbox!=None:

					if test[5]==label(str(minbox[4])):
						Count[0]+=1
					else:
						Count[1]+=1
				else:
					Count[1]+=1

				if minbox!=None:
					if test[5]==label(str(minbox[4])):
						confusionMatrix[test[5]][label(str(minbox[4]))]+=2
					else:
						confusionMatrix[test[5]][label(str(minbox[4]))]+=1

					a = 0
					b = 0
					if test[5]==0:
						a=0
					elif test[5]<3:
						a=1
					elif test[5]<4:
						a=2
					else:
						a=3
					if label(str(minbox[4]))==0:
						b=0
					elif label(str(minbox[4]))<3:
						b=1
					elif label(str(minbox[4]))<4:
						b=2
					else:
						b=3
					if a==b:
						confusionMatrix2[a][b]+=2
					else:
						confusionMatrix2[a][b]+=1

					if a!=3 and b!=3:
						if a==b:
							confusionMatrix3[a][b]+=2
						else:
							confusionMatrix3[a][b]+=1


					if test[5]==label("Person"):
						if str(minbox[4])=="Person":
							personCount[0]+=1
						else:
							personCount[1]+=1
					else: 
						if str(minbox[4])=="Person":
							personCount[2]+=1
						else:
							personCount[3]+=1

					if test[5]==label("Motorcycle"):
						if str(minbox[4])=="Motorcycle":
							motorcycleCount[0]+=1
						else:
							motorcycleCount[1]+=1
					else: 
						if str(minbox[4])=="Motorcycle":
							motorcycleCount[2]+=1
						else:
							motorcycleCount[3]+=1

					if test[5]==label("Bicycle"):
						if str(minbox[4])=="Bicycle":
							bicycleCount[0]+=1
						else:
							bicycleCount[1]+=1
					else: 
						if str(minbox[4])=="Bicycle":
							bicycleCount[2]+=1
						else:
							bicycleCount[3]+=1

					if test[5]==label("Car"):
						if str(minbox[4])=="Car":
							carCount[0]+=1
						else:
							carCount[1]+=1
					else: 
						if str(minbox[4])=="Car":
							carCount[2]+=1
						else:
							carCount[3]+=1

					if test[5]==label("Rickshaw"):
						if str(minbox[4])=="Rickshaw":
							rickshawCount[0]+=1
						else:
							rickshawCount[1]+=1
					else: 
						if str(minbox[4])=="Rickshaw":
							rickshawCount[2]+=1
						else:
							rickshawCount[3]+=1

					if test[5]==label("Autorickshaw"):
						if str(minbox[4])=="Autorickshaw":
							autorickshawCount[0]+=1
						else:
							autorickshawCount[1]+=1
					else: 
						if str(minbox[4])=="Autorickshaw":
							autorickshawCount[2]+=1
						else:
							autorickshawCount[3]+=1

					if test[5]==label("Autorickshaw") or test[5]==label("Rickshaw"):
						if str(minbox[4])=="Autorickshaw" or str(minbox[4])=="Rickshaw":
							threewheelerCount[0]+=1
						else:
							threewheelerCount[1]+=1
					else: 
						if str(minbox[4])=="Autorickshaw" or str(minbox[4])=="Rickshaw":
							threewheelerCount[2]+=1
						else:
							threewheelerCount[3]+=1

					if test[5]==label("Bicycle") or test[5]==label("Motorcycle"):
						if str(minbox[4])=="Bicycle" or str(minbox[4])=="Motorcycle":
							twowheelerCount[0]+=1
						else:
							twowheelerCount[1]+=1
					else: 
						if str(minbox[4])=="Bicycle" or str(minbox[4])=="Motorcycle":
							twowheelerCount[2]+=1
						else:
							twowheelerCount[3]+=1
				# else:
				# else:
				# 	Count[1]+=1
				# print maxratio, mindist
	print '\n\n',algo
	print count
	print 'Detection (Overlap)',count[0]/count[1]
	print 'Detection (Ratio)', ratioCount[0]/ratioCount[1]
	print 'Classification',Count[0]/(Count[1]+Count[0])
	print 'Person Classification', (personCount[0]+personCount[3])/(personCount[0]+personCount[1]+personCount[2]+personCount[3]), personCount
	print 'Motorcycle Classification', (motorcycleCount[0]+motorcycleCount[3])/(motorcycleCount[0]+motorcycleCount[1]+motorcycleCount[2]+motorcycleCount[3]), motorcycleCount
	print 'Bicycle Classification', (bicycleCount[0]+bicycleCount[3])/(bicycleCount[0]+bicycleCount[1]+bicycleCount[2]+bicycleCount[3]), bicycleCount
	print 'Car Classification', (carCount[0]+carCount[3])/(carCount[0]+carCount[1]+carCount[2]+carCount[3]), carCount
	print 'Rickshaw Classification', (rickshawCount[0]+rickshawCount[3])/(rickshawCount[0]+rickshawCount[1]+rickshawCount[2]+rickshawCount[3]), rickshawCount
	print 'Autorickshaw Classification', (autorickshawCount[0]+autorickshawCount[3])/(autorickshawCount[0]+autorickshawCount[1]+autorickshawCount[2]+autorickshawCount[3]), autorickshawCount
	print 'Two Wheeler Classification', (twowheelerCount[0]+twowheelerCount[3])/(twowheelerCount[0]+twowheelerCount[1]+twowheelerCount[2]+twowheelerCount[3]), twowheelerCount
	print 'Three Wheeler Classification', (threewheelerCount[0]+threewheelerCount[3])/(threewheelerCount[0]+threewheelerCount[1]+threewheelerCount[2]+threewheelerCount[3]), threewheelerCount
	print 'Classification Confusion Matrix'
	for i in confusionMatrix:
		print i
	print 1.0*np.trace(np.array(confusionMatrix))/np.sum(np.array(confusionMatrix))
	print 'Classification Confusion Matrix'
	for i in confusionMatrix2:
		print i
	print 1.0*np.trace(np.array(confusionMatrix2))/np.sum(np.array(confusionMatrix2))
	for i in confusionMatrix3:
		print i
	print 1.0*np.trace(np.array(confusionMatrix3))/np.sum(np.array(confusionMatrix3))
	# count = [0,0]
	# for i,j in zip(dump,pkl):
	# 	testframes = [item[0] for item in testingData[j]]
	# 	for frame in trainingData[i]:
	# 		if trainingData[i][frame[2]]-trainingData[i][frame][0] > 1000 or trainingData[i][frame][3]-trainingData[i][frame][1]>1000:
	# 			continue
	# 		if int(frame) not in testframes:
	# 			count[0]+=1
	# 		else:
	# 			count[1]+=1
	# print count
