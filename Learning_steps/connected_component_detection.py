
global input_folder,output_folder,input_pdf,input_file,input_for_extracting_JSON,temp_folder,input_image_from_pdf
global EXPECTED_X_LENGTH,EXPECTED_Y_LENGTH,Y_scale,X_scale,GRID_BEGIN_Y_AXIS,GRID_END_Y_AXIS,SEARCH_STARTING_POINT_SIGNAL
global SIGNAL_UPPER_LIMIT_FROM_CENTER,SIGNAL_LOWER_LIMIT_FROM_CENTER
global index_top_down
global lead_names
#global overlap_flag
#global df_merged
global index_gap_between_signals
#df_merged = pd.DataFrame(np.nan, index=list(range(0, 0)), columns=[])

#print "main"
# Pixel count of the PDF
EXPECTED_X_LENGTH = 3300
EXPECTED_Y_LENGTH = 2550
# Actual Size of the PDF
PDF_DIM_X = 792
PDF_DIM_Y = 612

Y_scale = 118.0 #    may not be significant as this will be normalized later on
global X_scale_obtained_from_PDF
X_scale_obtained_from_PDF = 1
X_scale = (25 * 12.0)
GRID_BEGIN_Y_AXIS = 650 #700
GRID_END_Y_AXIS = EXPECTED_Y_LENGTH
SEARCH_STARTING_POINT_SIGNAL = 65 #55
    # 170 255
SIGNAL_UPPER_LIMIT_FROM_CENTER = 120     # if 4 waves present
SIGNAL_LOWER_LIMIT_FROM_CENTER = 120     # if 4 waves present

index_top_down = []
#df_merged = []

lead_names = [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"],
             ["III", "aVF", "V3", "V6"]]
#####################################################################


#################################

import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import io
import sys
from os import listdir
from os.path import isfile, join

#image = cv2.imread('blob.jpg')
#image = cv2.imread('test3.jpg')
#image = cv2.imread('sample1.jpg')
#image = cv2.imread('blob.png')
#image=255-image

def extract_connected_components(image):
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#    cv2.imshow("image",thresh)
    # find contours in the thresholded image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    	cv2.CHAIN_APPROX_SIMPLE)[-2]
    print("[INFO] {} unique contours found".format(len(cnts)))
#    extracted_wave_image=image
    extracted_wave_image=255-(image*0)
    # loop over the contours
    for (i, c) in enumerate(cnts):
    	# draw the contour
    	if len(c)>700 :
                	((x, y), _) = cv2.minEnclosingCircle(c)
                	print len(c)
                	cv2.putText(image, "#{}".format(i + 1), (int(x) - 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                	cv2.drawContours(extracted_wave_image, [c], -1, (0, 0, 0), 2)
    cv2.imshow("Connected components",extracted_wave_image)
    cv2.imwrite("test_writing.jpg",extracted_wave_image)
    cv2.waitKey(0)
    return extracted_wave_image

def find_the_cut_between_signals():
#    print "find_the_cut_between_signals"
#    thresh1 = image_enhancement(img)  # function call for thresholding
    print image.shape
    thresh1 = extract_connected_components(image)
    print thresh1.shape

    #cv2.imshow('Threshholded .... image', thresh1)
    #cv2.waitKey(0)
    sum_row_to_find_cut = np.sum(thresh1, axis=1)
    sum_row_to_find_cut = (EXPECTED_X_LENGTH * 255) - sum_row_to_find_cut
    sum_part = sum_row_to_find_cut[0:thresh1.shape[1]]
    plt.plot(sum_part)
    plt.show()
    cv2.waitKey(0)
    for i in range(0, 3):
        index_gap = []
        for index in range(index_top_down[i], index_top_down[i + 1]):
            if sum_row_to_find_cut[index] == 0:
                index_gap.append(index)
                if np.mean(index_gap) == []:
                    print("Over lap .............")
                    global overlap_flag
                    overlap_flag = 1
#                    break
        global index_gap_between_signals
#        print("index_gap :")
#        print(index_gap)
#        print("index_gap_mean :" )
#        print( np.mean(index_gap))
#        print("index_gap_mean_ceil :" )
#        print( np.ceil(np.mean(index_gap)))
        index_gap_between_signals.append(np.ceil(np.mean(index_gap)))


#######################   main  ##################
#image = cv2.imread('blob.png')
#image=255-image
input_folder = sys.argv[1]
print input_folder
input_pdf = os.path.join(input_folder)
input_image="/Desktop/TEST_input/test1.jpg"
image = cv2.imread(input_pdf)
find_the_cut_between_signals()
#extract_connected_components(image)

