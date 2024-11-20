
import cv2

im = imread("test.jpg")
im_gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
axis("off")
title("Input Image")
imshow(im_gray, cmap = 'gray')
show()

retval, dst = cv2.threshold(src, thresh, maxval, type[, dst])


retval, im_at_fixed = cv2.threshold(im_gray, 50, 255, cv2.THRESH_BINARY)
axis("off")
title("Fixed Thresholding")
imshow(im_at_fixed, cmap = 'gray')
show()

dst=cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C[, dst])

im_at_mean = cv2.adaptiveThreshold(im_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 10)
axis("off")
title("Adaptive Thresholding with mean weighted average")
imshow(im_at_mean, cmap = 'gray')
show()


dst=cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C[, dst])

im_at_gauss = cv2.adaptiveThreshold(im_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 7)
axis("off")
title("Adaptive Thresholding with gaussian weighted average")
imshow(im_at_gauss, cmap = 'gray')
show()