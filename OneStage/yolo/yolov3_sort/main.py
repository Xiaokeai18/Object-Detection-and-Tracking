# import the necessary packages
import numpy as np
import argparse
import imutils
import time
import cv2
import os
import glob
import math

ref_point = (0,0)
cv2.namedWindow("output")
def point_selection(event, x, y, flags, param):
	# grab references to the global variables
	global ref_point
	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		ref_point = (x, y)
		cv2.circle(frame, ref_point, 2, (0, 255, 255), 2)

	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		ref_point = (x, y)
		# draw a rectangle around the region of interest
		cv2.circle(frame, ref_point, 2, (0, 255, 0), 2)
cv2.setMouseCallback("output", point_selection)

files = glob.glob('output/*.png')
for f in files:
   os.remove(f)

from sort import *
tracker = Sort()
memory = {}

counter = 0

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input",
	help="path to input video", default = "./input/det_t1_video_00031_test.avi")
ap.add_argument("-o", "--output",
	help="path to output video", default = "./output/")
ap.add_argument("-y", "--yolo",
	help="base path to YOLO directory", default = "./yolo-obj")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
ap.add_argument("-t", "--threshold", type=float, default=0.3,
	help="threshold when applyong non-maxima suppression")
args = vars(ap.parse_args())

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
	return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
	return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

# load the COCO class labels our YOLO model was trained on
labelsPath = os.path.sep.join([args["yolo"], "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

# initialize a list of colors to represent each possible class label
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(200, 3),
	dtype="uint8")

# derive the paths to the YOLO weights and model configuration
weightsPath = os.path.sep.join([args["yolo"], "yolov3.weights"])
configPath = os.path.sep.join([args["yolo"], "yolov3.cfg"])

# load our YOLO object detector trained on COCO dataset (80 classes)
# and determine only the *output* layer names that we need from YOLO
print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# initialize the video stream, pointer to output video file, and
# frame dimensions
vs = cv2.VideoCapture(args["input"])
writer = None
(W, H) = (None, None)

frameIndex = 0

# try to determine the total number of frames in the video file
try:
	prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
		else cv2.CAP_PROP_FRAME_COUNT
	total = int(vs.get(prop))
	print("[INFO] {} total frames in video".format(total))

# an error occurred while trying to determine the total
# number of frames in the video file
except:
	print("[INFO] could not determine # of frames in video")
	print("[INFO] no approx. completion time can be provided")
	total = -1
framecnt = 0
# loop over frames from the video file stream
while True:
	# read the next frame from the file
	(grabbed, frame) = vs.read()
	
	# if the frame was not grabbed, then we have reached the end
	# of the stream
	if not grabbed:
		break
	
	frame = cv2.resize(frame,(frame.shape[1]*480//frame.shape[0],480))
	
	if framecnt<0:
		framecnt += 1
		continue
	while ref_point==(0,0):
		cv2.imshow("output",frame)
		if cv2.waitKey(1) & 0xFF == 27: # Exit condition
			break
	# if the frame dimensions are empty, grab them
	if W is None or H is None:
		(H, W) = frame.shape[:2]

	currentxy =  [0,H,0,W]
	lefty,leftx,righty,rightx = 10000,10000,0,0
	# construct a blob from the input frame and then perform a forward
	# pass of the YOLO object detector, giving us our bounding boxes
	# and associated probabilities
	blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	net.setInput(blob)
	start = time.time()
	layerOutputs = net.forward(ln)
	end = time.time()

	# initialize our lists of detected bounding boxes, confidences,
	# and class IDs, respectively
	boxes = []
	confidences = []
	classIDs = []

	# loop over each of the layer outputs
	for output in layerOutputs:
		# loop over each of the detections
		for detection in output:
			# extract the class ID and confidence (i.e., probability)
			# of the current object detection
			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			# filter out weak predictions by ensuring the detected
			# probability is greater than the minimum probability
			if confidence > args["confidence"] and classID==0:
				# scale the bounding box coordinates back relative to
				# the size of the image, keeping in mind that YOLO
				# actually returns the center (x, y)-coordinates of
				# the bounding box followed by the boxes' width and
				# height
				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				# use the center (x, y)-coordinates to derive the top
				# and and left corner of the bounding box
				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				# update our list of bounding box coordinates,
				# confidences, and class IDs
				boxes.append([x, y, int(width), int(height)])
				confidences.append(float(confidence))
				classIDs.append(classID)

	# apply non-maxima suppression to suppress weak, overlapping
	# bounding boxes
	idxs = cv2.dnn.NMSBoxes(boxes, confidences, args["confidence"], args["threshold"])

	dets = []
	if len(idxs) > 0:
		# loop over the indexes we are keeping
		for i in idxs.flatten():
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			dets.append([x, y, x+w, y+h, confidences[i]])

	np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
	dets = np.asarray(dets)
	tracks = tracker.update(dets)

	boxes = []
	indexIDs = []
	c = []
	previous = memory.copy()
	memory = {}

	for track in tracks:
		boxes.append([track[0], track[1], track[2], track[3]])
		indexIDs.append(int(track[4]))
		memory[indexIDs[-1]] = boxes[-1]

	if len(boxes) > 0:
		i = int(0)
		for box in boxes:
			# extract the bounding box coordinates
			(x, y) = (int(box[0]), int(box[1]))
			(w, h) = (int(box[2]), int(box[3]))

			lefty = min(lefty,y)
			leftx = min(leftx,x)
			righty = max(righty,x+w)
			rightx = max(rightx,y+h)
			# draw a bounding box rectangle and label on the image
			# color = [int(c) for c in COLORS[classIDs[i]]]
			# cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

			color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
			cv2.rectangle(frame, (x, y), (w, h), color, 1)

			p0 = (int(x + (w-x)/2), int(y + (h-y)/2))
			if indexIDs[i] in previous:
				previous_box = previous[indexIDs[i]]
				(x2, y2) = (int(previous_box[0]), int(previous_box[1]))
				(w2, h2) = (int(previous_box[2]), int(previous_box[3]))
				p1 = (int(x2 + (w2-x2)/2), int(y2 + (h2-y2)/2))
				cv2.line(frame, p0, p1, color, 3)


				#counter += 1
			cv2.line(frame, p0, ref_point, color, 1)
			distance = math.sqrt((ref_point[0]-p0[0])**2+(ref_point[1]-p0[1])**2)
			gain = distance//20
			# text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
			text = "{}, +{:.0f}db".format(indexIDs[i],gain)
			cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
			i += 1
		
		if righty == 0:
			targetxy = [0,int(H),0,int(W)]
		else:
			scale = W/H
			if (rightx-leftx)/(righty-lefty)>scale:
				height_ = (rightx-leftx)*H//W 
				targetxy = [max((righty+lefty-height_),0)//2,max((righty+lefty-height_),0)//2+height_,leftx,rightx]
			else:
				width_ = (righty-lefty)*W//H
				targetxy = [lefty,righty,max((rightx+leftx-width_)//2,0),max((rightx+leftx-width_)//2,0)+width_] 

		for j in range(4):
				currentxy[j] += (targetxy[j]-currentxy[j])//1
		frame = cv2.resize(frame[currentxy[0]:currentxy[1],currentxy[2]:currentxy[3]],(frame.shape[1]*480//frame.shape[0],480))

		

	# draw counter
	cv2.putText(frame, "num: "+str(i), (0,70), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 255), 1)
	# counter += 1

	# saves image file
	#cv2.imwrite("output/frame-{}.png".format(frameIndex), frame)
	
	cv2.imshow("output",frame)
	if cv2.waitKey(1) & 0xFF == 27: # Exit condition
		break
	

	# check if the video writer is None
	if writer is None:
		# initialize our video writer
		fourcc = cv2.VideoWriter_fourcc(*"MJPG")
		writer = cv2.VideoWriter(args["output"], fourcc, 30,
			(frame.shape[1], frame.shape[0]), True)

		# some information on processing single frame
		if total > 0:
			elap = (end - start)
			print("[INFO] single frame took {:.4f} seconds".format(elap))
			print("[INFO] estimated total time to finish: {:.4f}".format(
				elap * total))

	# write the output frame to disk
	writer.write(frame)

	# increase frame index
	frameIndex += 1

	if frameIndex >= 9000:
		print("[INFO] cleaning up...")
		writer.release()
		vs.release()
		exit()

# release the file pointers
print("[INFO] cleaning up...")
if not writer:
	writer.release()
vs.release()
