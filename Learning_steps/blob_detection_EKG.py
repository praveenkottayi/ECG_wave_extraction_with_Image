# Standard imports
import cv2
import numpy as np;

# Read image
img1 = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)

ret, thresh1 = cv2.threshold(img1, 50, 255, cv2.THRESH_BINARY)
cv2.imshow("thrs", thresh1)
# Set up the detector with default parameters.
detector = cv2.SimpleBlobDetector()

# Detect blobs.
keypoints = detector.detect(thresh1)
print keypoints
# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(thresh1, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show keypoints
cv2.imshow("Keypoints", im_with_keypoints)
cv2.waitKey(0)



