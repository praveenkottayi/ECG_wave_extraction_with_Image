import cv2
import numpy as np
import pdfrw
from matplotlib import pyplot as plt


img = cv2.imread('test.jpg', 0)
print("heelo")
#img = cv2.imread('Brigham_Sample_EKG_full.jpg', 0)
cv2.imshow('image', img)
cv2.waitKey(0)
hist = cv2.calcHist([img],[0],None,[256],[0,256])
plt.hist(img.ravel(),256,[0,256]);
plt.show()

#print img
ret, thresh1 = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)
print(ret)
cv2.imshow('EKG_processed_wave', thresh1)
hist = cv2.calcHist([thresh1],[0],None,[256],[0,256])
plt.hist(thresh1.ravel(),256,[0,256]);
plt.show()

#######################################################################

blur = cv2.GaussianBlur(img,(5,5),0)

# find normalized_histogram, and its cumulative distribution function
hist = cv2.calcHist([blur],[0],None,[256],[0,256])
hist_norm = hist.ravel()/hist.max()
Q = hist_norm.cumsum()

bins = np.arange(256)

fn_min = np.inf
thresh = -1

for i in xrange(1,256):
    p1,p2 = np.hsplit(hist_norm,[i]) # probabilities
    q1,q2 = Q[i],Q[255]-Q[i] # cum sum of classes
    b1,b2 = np.hsplit(bins,[i]) # weights

    # finding means and variances
    m1,m2 = np.sum(p1*b1)/q1, np.sum(p2*b2)/q2
    v1,v2 = np.sum(((b1-m1)**2)*p1)/q1,np.sum(((b2-m2)**2)*p2)/q2

    # calculates the minimization function
    fn = v1*q1 + v2*q2
    if fn < fn_min:
        fn_min = fn
        thresh = i

print(thresh)
# find otsu's threshold value with OpenCV function
ret, thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print(thresh,ret)

hist = cv2.calcHist([thresh],[0],None,[256],[0,256])
plt.hist(thresh.ravel(),256,[0,256]); plt.show()


cv2.imshow('EKG_processed_wave_auto_threshold', thresh)
cv2.waitKey(0)

#########################################################################
