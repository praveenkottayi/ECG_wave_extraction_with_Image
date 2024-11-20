import numpy as np
import PythonMagick
import cv2
import matplotlib.pyplot as plt
############################################################
print " Reading PDF "
img = PythonMagick.Image()
img.density("300")

try:
#    img.read("input/sample 1.pdf") #  read in at 300 dpi


#    img.write("Brigham_Sample_converted_image.jpg")
#    img = cv2.imread('blob.png',0)
    img = cv2.imread('sample1.jpg',0)
#    img = cv2.imread('sample1_simulated_peak_2.jpg',0)
    ret, thresh = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

    print thresh.shape
    cv2.imshow('Converted image', thresh)
    sum_row_to_find_cut = np.sum(thresh, axis=1)
#    sum_row_to_find_cut = (thresh.shape[0] * 255) - sum_row_to_find_cut
    sum_part = sum_row_to_find_cut[0:thresh.shape[1]]
#    print sum_part
    plt.plot(sum_part.T)
    plt.show()
    k = cv2.waitKey(0)

    if k == 27:         # wait for ESC key to exit
        cv2.destroyAllWindows()
    elif k == ord('s'): # wait for 's' key to save and exit
        cv2.destroyAllWindows()

except Exception as e:
    print(" no such file")
    print e

print" mission accomplished . Go and check folder"
############################################################
