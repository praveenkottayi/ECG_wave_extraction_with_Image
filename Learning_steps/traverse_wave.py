##########################################################################
#                   EKG Header and Signal Extractor
#
#  @ Praveen
#  @ 13th April 2016
#
# The program :
# 1. Extracts Header from the PDF and save as JSON (With name as of ID field in the EKG file)
# 2. Extarcts the signals and saves as CSV (With name as of ID field in the EKG file)
##########################################################################

# Libraries used


import logging
import numpy as np
import cv2
import pandas as pd
#import array
import matplotlib.pyplot as plt
#import PyPDF2
import PythonMagick
import os
import re
import json
#from json import dumps, load
import io
import sys
from os import listdir
from os.path import isfile, join
##########################################################################

# For extracting one row of signal wave if predified boundaries are given

def extract_one_row(i):
    x1 = index_top_down[i]
    x2 = x1
    y1 = 0
    y2 = EXPECTED_X_LENGTH
    img1 = img[x1 - SIGNAL_UPPER_LIMIT_FROM_CENTER:
        x2 + SIGNAL_LOWER_LIMIT_FROM_CENTER, y1:y2]
#    cv2.imshow(str(i),img1)
    thresh1 = image_enhancement(img1)  # function call for thresholding
#    print "size of thrshld" + str(thresh1.shape)
    ecg_wave = digitize_full_row_ecg(thresh1)
    index_split = split_EKG_into_leads(ecg_wave)
    ecg_wave_normalized = normalize_EKG(ecg_wave)
    cv2.waitKey(0)
    return (ecg_wave_normalized, index_split)


##########################################################################

# For extracting one row of signal wave if algorithm finds the boundary

def extract_one_row_automatic(i):
#    print "extract_one_row_automatic"
    global index_gap_between_signals
    y1 = int(index_gap_between_signals[i])
    y2 = int(index_gap_between_signals[i + 1])
    x1 = 0
    x2 = EXPECTED_X_LENGTH
    img1 = img[y1:y2, x1:x2]

    cv2.imshow(str(i),img1)
    thresh1 = image_enhancement(img1)  # function call for thresholding
#    print(("size of thrshld" + str(thresh1.shape)))
#    print("............................................")
#    print((thresh1.shape))
#    print((sum(thresh1[1, :])))
#    print("............................................")
    ecg_wave = digitize_full_row_ecg(thresh1)
    index_split = split_EKG_into_leads(ecg_wave)
    ecg_wave_normalized = normalize_EKG(ecg_wave)
    cv2.waitKey(0)
    return (ecg_wave_normalized, index_split)


##########################################################################

# For enhancing the image (part of siganl /Whole image ), mainly thresholding

def image_enhancement(img1):
#    print "image_enhancement"
    ret, thresh1 = cv2.threshold(img1, 50, 255, cv2.THRESH_BINARY)
#    blur = cv2.GaussianBlur(img,(5,5),0)
#    ret, thresh1 = cv2.threshold(blur,160,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#    print ret
#    cv2.imshow('EKG_processed_wave', thresh1)
#    cv2.waitKey(0)
    return thresh1

########################  Digitize wave signal  ##########################

# For extracting the signal wave from the image to number array based on pixels

def digitize_full_row_ecg(thresh):
#    print "digitize_full_row_ecg"
#    xxx = [None] * thresh.shape[1]
    ecg_wave = [None] * thresh.shape[1]
    X_range = thresh.shape[0]
    for y in range(1, thresh.shape[1]):
        for x in range(1, X_range):
            if thresh[x, y] == 0:
                ecg_wave[y] = X_range - x
                break
    return ecg_wave

# For identifying 4 leads from the signal

def split_EKG_into_leads(ecg_wave):
#    print "split_EKG_into_leads"
    index_break_wave = [item for item in range(len(ecg_wave)) if ecg_wave[item] == None]
#    print index_break_wave
    index_L_R = []
    for item in range(1, len(index_break_wave) - 1):
        difference = index_break_wave[item] - index_break_wave[item + 1]
        if (-1 * difference > 100):
            start_of_wave = index_break_wave[item]
            index_L_R.append(start_of_wave)
    index_L_R.append(index_break_wave[len(index_break_wave) - 1])
    return index_L_R

# For normalizing the wave and making it centered at zero

def normalize_EKG(ecg_wave):
#    print "normalize_EKG"
    ecg_wave_normalized = ecg_wave
    for item in range(index_top_down[0], index_top_down[1]):
        if (ecg_wave[item] > 10):
            norm_value = ecg_wave[item]
            break
#    print((norm_value))
    for item in range(1, len(ecg_wave)):
        cond = ecg_wave[item]
        if cond is not None:
            ecg_wave_normalized[item] = ecg_wave[item] - norm_value
    return ecg_wave_normalized

##########################################################################

# For visual inspection of how the waves are splitted and final csv is returned

def plot_splitted_EKG(i, a, b, index, ecg_wave_normalized):
#    print("plot_splitted_EKG")
    df = pd.DataFrame(np.nan, index=list(range(index[a], index[b])), columns=[])
    df['Lead'] = "Lead" + " " + lead_names[i][a]
    df['absoluteX'] = range(index[a], index[b])
    df['absoluteY'] = ecg_wave_normalized[index[a]:index[b]]
    df['actual_X'] = [float(x / X_scale) for x in range(index[a], index[b])]
    df['actual_Y'] = [float(x / Y_scale) for x in df['absoluteY']]

#    print(("size of df " + str(len(df))))
    global df_merged
    df_merged = df_merged.append(df) #  From df_merged the final CSV is generated
#    logger.error("plot splitted EKG")
#    logger.error(len(df_merged))
#    df.to_csv('df_ ' + str(i) + '.csv')     # Save as CSV

    plt.plot(ecg_wave_normalized[index[a]:index[b]])
    plt.title('Extracted wave ' + lead_names[i][a])
    plt.xlabel('time')
    plt.ylabel('mV')
    plt.show()

############################################################################

# For finding the gap between two signals and check if they are overlapping

def find_the_cut_between_signals():
    print "find_the_cut_between_signals"
    thresh1 = image_enhancement(img)  # function call for thresholding
    #cv2.imshow('Threshholded .... image', thresh1)
    #cv2.waitKey(0)
    sum_row_to_find_cut = np.sum(thresh1, axis=1)
    sum_row_to_find_cut = (EXPECTED_X_LENGTH * 255) - sum_row_to_find_cut
#    sum_part = sum_row_to_find_cut[700:index_top_down[4]]
#    plt.plot(sum_part)
#    plt.show()
    print "........"
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
        print "..............................................."
        print index_gap_between_signals
        print "end of find_the_cut_between_signals"


########################  Extract text from PDF #############

# For extracting the header contents from the PDF

def extract_EKG_Header(input_file):
#    print "extract_EKG_Header"
    print (" PDF reading in progress")
#    newdir = "input"
    os.chdir(os.path.expanduser(input_folder))
#    print(input_file)
    #os.path.isfile(input_file)
    os.system(''.join("pdftotext " + input_file + " -layout"))

    #################### Extract from text #################

    file_header = open(input_for_extracting_JSON, 'r').readlines()
    for i in file_header:
        thisline = list(i)
        for j in range(1, len(i) - 2):
            first = i[j]
            second = i[j + 1]
            third = i[j + 2]
            if first != ' ' and second == ' ' and third != ' ':
                thisline[j + 1] = '$'
        str1 = ''.join(thisline)
    #    print str1 ############################
        words = str1.split()
#        print len(words)

        for i in range(0, len(words)):
            # print the word
            if "Ve" in words[i]:
                Vent_Rate = words[i + 1]
            if "PR" in words[i]:
                PR_Interval = words[i + 1]
            if "QR" in words[i]:
                QRS_Duration = words[i + 1]
            if "QT/" in words[i]:
                QT_QTc = words[i + 1]
            if "P-R-T" in words[i]:
#                PRT = words[i + 1]
                PRT =  words[i+1].replace('$',';') + ';' + (words[i+2])
#                print words[i+2]
            if "BP" in words[i] and "BPM" not in words[i]:
                BP = words[i + 1]
            if "mm/s" in words[i]:
                SCALE_X = words[i]
                X_scale_obtained_from_PDF =int(re.sub('[^0-9]', '', SCALE_X))
                global X_scale_obtained_from_PDF
            if "mm/mV" in words[i]:
                SCALE_Y = words[i]
            if "Hz" in words[i]:
                SIGNAL = words[i]
            if "ID" in words[i] and "CID"not in words[i] and "EID"not in words[i]:
                print((words[i]))
                ID = words[i]
                Name = words[i - 1]
                Date = words[i + 1]
            if "Male" in words[i]:
                Sex = words[i]
            if "Female" in words[i]:
                Sex = words[i]
            if "yr" in words[i]:
                DOB = words[i]
            if "ACC" in words[i]:
                ACCOUNT = words[i]

#            if "Room" in words[i]:
#                print(words[i])
#            if "Loc" in words[i]:
#                print(words[i])
#            if "Confirmed" in words[i]:
#                print(words[i])
#            if "Referred" in words[i]:
#                print words[i]
#            if "Tech" in words[i]:
#                print(words[i])

    BP = " "
    ACCOUNT = "$:"

    data = {"Vent_Rate" : Vent_Rate, "PR_Interval" : PR_Interval, "QRS_Duration" : QRS_Duration, "QT_QTc":QT_QTc, "PRT":PRT.replace('$', ' '), "BP":BP, "SCALE_X":SCALE_X,
        "SCALE_Y":SCALE_Y, "SIGNAL":SIGNAL, "ID":ID.split(':')[1], "Name":Name.replace('$',' '), "Date":Date.replace('$',' '), "Sex":Sex, "DOB":DOB.replace('$', ' '), "ACCOUNT":ACCOUNT.replace('$', ' ').replace('$', ' ').split(':')[1]}  # lint:ok

    ID_as_filename = ''.join(str(ID.split(':')[1]))
    append_name = input_file.split('.')[0].replace('/', '_')
    ID_as_filename = join(ID_as_filename + '_' + append_name)
#    print(".................................................")
#    print append_name
#    print(ID_as_filename)
#    print(".................................................")
    file_json = ''.join(ID_as_filename + ".json")
    file_json = os.path.join(output_folder, file_json)
    with io.open(file_json, 'w', encoding='utf-8') as f:
        f.write(unicode(json.dumps(data, ensure_ascii=False)))
    return ID_as_filename
    print(" PDF reading in completed")


# Function to fetch CSV and the header , from here all other functions are called

def getSignalCSVandJSONfromPdf(read_image):
#    print "getSignalCSVandJSONfromPdf"


    # If image is in predefined dimension
    img= cv2.imread(read_image,0)
    if img.shape[0] == EXPECTED_Y_LENGTH and img.shape[1] == EXPECTED_X_LENGTH:
        global df_merged
        df_merged = pd.DataFrame(np.nan, index=list(range(0, 0)), columns=[])

#        lead_names = [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"],
#             ["III", "aVF", "V3", "V6"]]
        base = [None] * (GRID_END_Y_AXIS)
        index_top_down = []
        global index_gap_between_signals
        index_gap_between_signals = [GRID_BEGIN_Y_AXIS]

        global overlap_flag
        overlap_flag = 0
        print GRID_BEGIN_Y_AXIS
        print GRID_END_Y_AXIS
        print "............ahaa............."
        print index_gap_between_signals
        print SEARCH_STARTING_POINT_SIGNAL
        print " look above"




        for x in range(GRID_BEGIN_Y_AXIS, GRID_END_Y_AXIS):
            for y in range(1, SEARCH_STARTING_POINT_SIGNAL):
                if img[x, y] == 0:
                    base[x] = y
                    index_top_down.append(x)
                    print index_top_down
                    break
        print base
        # this has the number of signal waves presnet in the grid
        test=[]
        print len(index_top_down)
        for index in range(0,len(index_top_down)-1):
        	temp1=index_top_down[index]
        	temp2=index_top_down[index+1]
#        	print temp2 -temp1
        	if (temp2-temp1)>50:
        	    test.append(temp1)
        test.append(temp1)
#        print test
#        print "changed.. index"
        global index_top_down
        index_top_down=test
        print "actaul index_top_down"
        print((index_top_down))

    ###############################################################
        try:
            find_the_cut_between_signals()
        except Exception as e:
            logger.error("Error in function to find overlap and cut between signals in %s" % (input_file))
            logger.error(e)
    ###############################################################

        if overlap_flag == 0:  # 0 ################### loop control
            iter_rows = 3  # number of rows to be extracted
            for i in range(0, iter_rows):
#                print((index_top_down[i]))

                E, I = extract_one_row_automatic(i)  # HI

                for j_index in range(0, 4):
                    plot_splitted_EKG(i, j_index, j_index + 1, I, E)
#                    print((str(i) + str(j_index)))
#            print((ID_as_filename))
    #        df_merged.to_csv(ID_as_filename + ".csv")  # Save as CSV
            logger.error(len(df_merged))
            df_merged.to_csv(os.path.join(output_folder, (ID_as_filename + ".csv")), index=False)  # Save as CSV
            os.remove(input_for_extracting_JSON)
            os.remove(input_image_from_pdf)
        else:
            logger.error("Signal Overlap : %s " % (input_file))



####################################################################






##########################################################################
#                       main
##########################################################################


#####################  SET Variables #####################################

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
GRID_BEGIN_Y_AXIS = 650
GRID_END_Y_AXIS = EXPECTED_Y_LENGTH
SEARCH_STARTING_POINT_SIGNAL = 65
    # 170 255
SIGNAL_UPPER_LIMIT_FROM_CENTER = 120     # if 4 waves present
SIGNAL_LOWER_LIMIT_FROM_CENTER = 120     # if 4 waves present

index_top_down = []
#df_merged = []

lead_names = [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"],
             ["III", "aVF", "V3", "V6"]]
#####################################################################

logger = logging.getLogger('MAIN')
hdlr = logging.FileHandler(os.path.join('EKG_EXTRACTION.log'))
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.WARNING)


#####################################################################

#  Converting input pdf to image (jpg with dpecified dpi) #########
input_pdf = "test case 2.jpg"
img = cv2.imread(input_pdf, 0)
#cv2.imshow('Original_wave', img)

thresholded_image=image_enhancement(img)
cv2.imshow('Thresholded_wave', thresholded_image)

print thresholded_image.shape
ecg_wave = digitize_full_row_ecg(thresholded_image)
#print ecg_wave
index_split = split_EKG_into_leads(ecg_wave)
print index_split

################## traversal ############################

print " ..........Traversal way........................."
traversal_wave = [None] * thresholded_image.shape[1]

mid= int(round(thresholded_image.shape[0]/2))


sums_in_each_vertical_line = [255*len(thresholded_image)-sum(row) for row in zip(*thresholded_image)]
print sums_in_each_vertical_line

traversed_wave_min=1000
for x_val in range (0,thresholded_image.shape[1]):
    for y_val in range (mid-100 ,mid+100):
        if thresholded_image[y_val,x_val]==0:
            start_x = x_val
            start_y = y_val
            break

print start_x
print start_y

the_traveresd_line=[None] * thresholded_image.shape[1]

for x_val in range (0,thresholded_image.shape[1]):
    for y_val in range (0 ,thresholded_image.shape[0]):
        if thresholded_image[y_val,x_val]==0 :
            # and abs(the_traveresd_line[x_val]-yval) < 1
            the_traveresd_line[x_val] = thresholded_image.shape[0]-y_val
            break
print start_x
print the_traveresd_line

plt.plot(the_traveresd_line)
plt.title('1st method' )
#plt.show()

the_traveresd_line_2=[None] * thresholded_image.shape[1]
print int(round(thresholded_image.shape[0]/2))

for x_val in range (0,thresholded_image.shape[1]):
    y_val=int(round(thresholded_image.shape[0]/2))
    while y_val < thresholded_image.shape[0]:
        if thresholded_image[y_val,x_val]==0 :
            the_traveresd_line_2[x_val] = abs(thresholded_image.shape[0]- y_val)
            y_val=y_val+1
            break
        y_val=y_val+1
plt.plot(the_traveresd_line_2)
plt.title('2nd method' )
#plt.show()

the_traveresd_line_3=[None] * thresholded_image.shape[1]

for x_val in range (0,thresholded_image.shape[1]):
    y_val=int(round(thresholded_image.shape[0]/2))
    while y_val > 0:
        # print y_val
        if thresholded_image[y_val,x_val] == 0 :
            the_traveresd_line_2[x_val] = 100 #abs(thresholded_image.shape[0]- y_val)
            y_val=y_val-1
            break
        y_val=y_val-1
plt.plot(the_traveresd_line_3)
plt.title('3rd method' )
#plt.show()

#print start_x
print the_traveresd_line_3

print "..................huui hoii .........."


#print ecg_wave
plt.plot(ecg_wave)
plt.title('Extracted wave' )
plt.xlabel('time')
plt.ylabel('mV')
#plt.show()


read_image="sample1_simulated_peak_1.jpg"
#read_image="sample4.jpg"
input_file=read_image
img=cv2.imread(read_image,0)
print img.shape
cv2.imshow(read_image,img)
getSignalCSVandJSONfromPdf(read_image)


#########################################################

#ecg_wave_normalized = normalize_EKG(ecg_wave)
cv2.waitKey(0)

#######################################################################
#                         END OF MAIN
#######################################################################

