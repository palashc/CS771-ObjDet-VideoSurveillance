import argparse
import datetime
import imutils
import time
import cv2
from sklearn.externals import joblib
import numpy as np

clf = joblib.load("ADA500.pkl")
km = joblib.load("SiftKMeans_mini_clf.pkl")
sift = cv2.SIFT(100)
p = len(np.unique(km.labels_))
detect = np.zeros((1,6))

def label(o):           #I have no idea what labels to give
	a = []
	if o=="Person":
		ans = 0
	elif o=="Motorcycle":
		ans = 1
	elif o=="Bicycle":
		ans = 2
	elif o=="Number-plate":
		ans = 3
	elif o=="Car":
		ans = -1
	elif o=="Rickshaw":
		ans = -2
	elif o=="Autorickshaw":
		ans = -3
	a.append(ans)
	return a

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-c", "--classifier", help="classifier")
ap.add_argument("-a", "--min-area", type=int, default=5000, help="minimum area size")
args = vars(ap.parse_args())

# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
	camera = cv2.VideoCapture(0)
	time.sleep(0.25)

# otherwise, we are reading from a video file
else:
	camera = cv2.VideoCapture(args["video"])

# initialize the first frame in the video stream
# firstFrame = None
firstFrame = cv2.GaussianBlur(cv2.cvtColor(cv2.imread('bg.jpg'), cv2.COLOR_BGR2GRAY), (21, 21), 0)
iframe = 0

# loop over the frames of the video
while True:
	# grab the current frame and initialize the occupied/unoccupied
	# text
	(grabbed, frame) = camera.read()

	# if the frame could not be grabbed, then we have reached the end
	# of the video
	if not grabbed:
		break


	frame = cv2.resize(frame,None,fx=0.5,fy=0.5, interpolation=cv2.INTER_CUBIC)
	# resize the frame, convert it to grayscale, and blur it
	# frame = imutils.resize(frame, width=500,height=500)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)


	# print firstFrame.shape,gray.shape
	# if the first frame is None, initialize it
	if firstFrame is None:
		firstFrame = gray
		continue

	# compute the absolute difference between the current frame and
	# first frame
	frameDelta = cv2.absdiff(firstFrame, gray)
	thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
	thresh = cv2.dilate(thresh, None, iterations=2)
	(cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)

	# loop over the contours
	for c in cnts:
		# if the contour is too small, ignore it
		if cv2.contourArea(c) < args["min_area"]:
			continue

		# compute the bounding box for the contour, draw it on the frame,
		# and update the text

		(x, y, w, h) = cv2.boundingRect(c)

		crop_image = frame[y:y+h, x:x+w]
		kp, des = sift.detectAndCompute(crop_image,None)

		if des == None:
			continue
		if len(des) < 20:
			continue


		a = np.zeros((p,))
		clusters = km.predict(des)

		for c in clusters:
			a[c] +=1

		mk = clf.predict(a).tolist()[0]
		d = np.zeros((1,6))
		d[0,0] = iframe
		d[0,1] = x
		d[0,2] = y
		d[0,3] = w
		d[0,4] = h
		d[0,5] = mk
		detect = np.concatenate((detect,d),axis=0)

		# print type(mk)
		Colours = {4:(255,0,0), 5:(255,0,0),0:(0,0,255),1:(255,0,0),2:(255,0,0),3:(255,0,0) }
		print mk
		# print Colours[mk]
		cv2.rectangle(frame, (x, y), (x + w, y + h), Colours[mk], 2)

	cv2.imwrite('bg/'+str(iframe)+'.jpg',frameDelta)
	iframe +=1
	# frame = cv2.resize(frame,None,fx=0.5,fy=0.5, interpolation=cv2.INTER_CUBIC)
	# frameDelta = cv2.resize(frameDelta,None,fx=0.5,fy=0.5, interpolation=cv2.INTER_CUBIC)
	cv2.imshow("Labelled Video", frame)
	# cv2.imshow("Thresh", thresh)
	cv2.imshow("Background Substraction", frameDelta)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break

detect = detect[1:]
detect.dump(args["video"]+"_bgfg.pkl")

camera.release()
cv2.destroyAllWindows()
