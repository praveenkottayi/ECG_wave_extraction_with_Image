# Standard imports
import cv2
import numpy as np;

# Read image
im = cv2.imread("blob.png", cv2.IMREAD_GRAYSCALE)
im = (255-im)
# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 0
params.maxThreshold = 200


# Filter by Area.
params.filterByArea = True
params.minArea = 2000

# Filter by Circularity
params.filterByCircularity = False
params.minCircularity = 0.9

# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.95

# Filter by Inertia
params.filterByInertia = False
params.minInertiaRatio = 0.8

# Create a detector with the parameters
ver = (cv2.__version__).split('.')
if int(ver[0]) < 3 :
	detector = cv2.SimpleBlobDetector(params)
else :
	detector = cv2.SimpleBlobDetector_create(params)


# Detect blobs.
keypoints = detector.detect(im)
print(keypoints)
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# the size of the circle corresponds to the size of blob

im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show blobs
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)