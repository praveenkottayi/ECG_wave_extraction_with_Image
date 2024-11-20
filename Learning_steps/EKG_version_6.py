##########################################################################
#                   EKG Header and Signal Extractor
#
#  @ Author    : Praveen V
#  @ Date      : 29th june 2016
#
# The program :
# 1. Extracts Header from the PDF and save as JSON (With name as of ID field in the EKG file)
# 2. Extarcts the signals and saves as CSV (With name as of ID field in the EKG file)
##########################################################################

"""EKG Header and Signal Extractor"""
##########################################################################
# Libraries used

import logging
import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import PyPDF2
import PythonMagick
import os
import re
import json
import io
import sys
from os import listdir
from os.path import isfile, join

##########################################################################

# For extracting one row of signal wave if algorithm finds the boundary

def extract_one_row_automatic(start,end):
    global index_gap_between_signals
    # x1,x2,y1,y2 are dimension of the image
    y1 = int(start)
    y2 = int(end)
    x1 = 0
    x2 = EXPECTED_X_LENGTH
    global thresh_original_image
    img1 = thresh_original_image[y1:y2, x1:x2]   # img
    thresh1 = extract_connected_components(img1)
    ecg_wave = digitize_full_row_ecg(thresh1)
    index_split = split_EKG_into_leads(ecg_wave)
    ecg_wave_normalized = normalize_EKG(ecg_wave)
    return (ecg_wave_normalized, index_split)


###############################################################################

# For enhancing the image (part of siganl /Whole image ), mainly thresholding
# This is critical as the incorrect thresholding can ruin the digitalization
# as well as the CSV conversion

def image_enhancement(img1):
    # Thresholding for MAC and Linux PC can vary.
    # Found better to go with Mac version values as that works well with linux also

    ret, thresh1 = cv2.threshold(img1, 50, 255, cv2.THRESH_BINARY)
#    blur = cv2.GaussianBlur(img,(5,5),0)
#    ret, thresh1 = cv2.threshold(blur,160,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#    print ret
#    cv2.imshow('EKG_processed_wave', thresh1)
#    cv2.waitKey(0)
    return thresh1

########################  Digitize wave signal  ###############################

# For extracting the signal wave from the image to number array based on pixels
# Algorithm works by scanning for black pixels from top

def digitize_full_row_ecg(thresh):
    ecg_wave = [None] * thresh.shape[1]
    X_range = thresh.shape[0]
    for y in range(1, thresh.shape[1]):
        for x in range(1, X_range):
            if thresh[x, y] == 0:
                ecg_wave[y] = X_range - x
                break
    return ecg_wave

###############################################################################
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

###############################################################################
# For normalizing the wave and making it centered at zero

def normalize_EKG(ecg_wave):
    ecg_wave_normalized = ecg_wave
    for item in range(index_top_down[0], index_top_down[1]):
        if (ecg_wave[item] > 10):
            norm_value = ecg_wave[item]
            break

    for item in range(1, len(ecg_wave)):
        cond = ecg_wave[item]
        if cond is not None:
            ecg_wave_normalized[item] = ecg_wave[item] - norm_value
    return ecg_wave_normalized

###############################################################################

# For visual inspection of how the waves are splitted and final csv is returned
# This is where on each iteration the dataframe gets appeneded with lead data
# the dataframe is written to the CSV in the main function

def plot_splitted_EKG(i, a, b, index, ecg_wave_normalized):
    df = pd.DataFrame(np.nan, index=list(range(index[a], index[b])), columns=[])
    df['Lead'] = "Lead" + " " + lead_names[i][a]
    df['absoluteX'] = range(index[a], index[b])
    df['absoluteY'] = ecg_wave_normalized[index[a]:index[b]]
    df['actual_X'] = [float(x / X_scale) for x in range(index[a], index[b])]
    df['actual_Y'] = [float(x / Y_scale) for x in df['absoluteY']]
#    print(("size of df " + str(len(df))))
    global df_merged
    df_merged = df_merged.append(df) #  From df_merged the final CSV is generated

#    df.to_csv('df_ ' + str(i) + '.csv')     # Save as CSV
#    plt.plot(ecg_wave_normalized[index[a]:index[b]])
#    plt.title('Extracted wave ' + lead_names[i][a])
#    plt.xlabel('time')
#    plt.ylabel('mV')
#    plt.show()

###############################################################################

# For finding the gap between two signals and check if they are overlapping
# First method to solve overlaping
# Here we find the gap between the signals for understanding the structure of EKG

def find_the_cut_between_signals():
    thresh1 = image_enhancement(img)  # function call for thresholding
    #cv2.imshow(file, thresh1)
    #cv2.waitKey(0)
    thresh1 = extract_connected_components(img)
    global thresh_original_image
    thresh_original_image=thresh1
    #cv2.imshow(file, thresh1)
    #cv2.waitKey(0)
    sum_row_to_find_cut = np.sum(thresh1, axis=1)
    sum_row_to_find_cut = (EXPECTED_X_LENGTH * 255) - sum_row_to_find_cut
    #print len(index_top_down)
    start_limit = 0

    # index_top_down contains how many rows of signal
    # We need only 3 rows ( as per earlier requirement)
    # the last line of the area of interest in EKG can be found by
    # 1) the 4rth row
    # 2) the last point of the EKG

    if len(index_top_down) >4:
    	end_limit = index_top_down[4]
    else :
	end_limit = EXPECTED_Y_LENGTH

    sum_part = sum_row_to_find_cut[start_limit:end_limit]
    #plt.plot(sum_part)
    #plt.show()

    # Remember a line in EKG is a collection of pixels. So we have to find
    # the cluster of pixels for finding where to split.

    for i in range(0, 3):
        index_gap = []
        for index in range(index_top_down[i], index_top_down[i + 1]):
            if sum_row_to_find_cut[index] == 0:
                index_gap.append(index)
                if len(index_gap)>=1 :
                    mean_index_gap = np.mean(index_gap)
                else :
                    mean_index_gap = []
                if mean_index_gap == []:
                    #print("Over lap .............")
                    global overlap_flag
                    overlap_flag = 1
                    break
        global index_gap_between_signals

        index_gap_between_signals.append(np.ceil(np.mean(index_gap)))

    # For handling the overlaping of the last line
    if np.isnan(index_gap_between_signals[3]) :
        index_gap_between_signals[3]=2500

#    print index_gap_between_signals
#    print index_top_down
    #plt.plot((index_gap_between_signals[0], index_gap_between_signals[0]), (0,np.max(sum_part) ), 'k-')
    #plt.plot((index_gap_between_signals[1], index_gap_between_signals[1]), (0,np.max(sum_part) ), 'k-')
    #plt.plot((index_gap_between_signals[2], index_gap_between_signals[2]), (0,np.max(sum_part) ), 'k-')
    #plt.plot((index_gap_between_signals[3], index_gap_between_signals[3]), (0,np.max(sum_part) ), 'k-')
    #plt.show()

    for i in range(0,3):
#        global index_gap_between_signals
        if np.isnan(index_gap_between_signals[i]):
            index_gap_between_signals[i]=index_top_down[i]
    #print "...........check 1 completed......"
    find_overlap(img)
    #print "...........check 2 completed......"

###############################################################################
# For detecting the case 5 ,  i.e to find overlap of signals in adjacent rows.
###############################################################################

def find_overlap(image):
    thresh1 = extract_connected_components(image)
    sum_row_to_find_cut = np.sum(thresh1, axis=1)
    sum_row_to_find_cut = (EXPECTED_X_LENGTH * 255) - sum_row_to_find_cut
    sum_part = sum_row_to_find_cut[0:thresh1.shape[1]]

    highest_signal_point = -1
    lowest_signal_point= -1
    middle_signal_point = -1

    for i in range (0,len(sum_row_to_find_cut)-1):
        if (np.mean(sum_row_to_find_cut[i])> 0):
            highest_signal_point=i
            break

    for i in range (0,len(sum_row_to_find_cut)-1):
        if (np.mean(sum_row_to_find_cut[len(sum_row_to_find_cut)-1-i]) > 0):
            lowest_signal_point=len(sum_row_to_find_cut)-1-i
            break

    for i in range (0,len(sum_row_to_find_cut)-1):
        if (np.mean(sum_row_to_find_cut[i]) ==  sum_row_to_find_cut.max()):
            middle_signal_point= i
    global count_overlap_files

    # If overlap is found then print those EKG file names to a file

    if (highest_signal_point < 2 or lowest_signal_point > len(sum_row_to_find_cut)):
        global input_file
        count_overlap_files = count_overlap_files +1
        with open("EKG_overlap_list.txt", 'a') as f:
        #f = open("EKG_Overlap_files.txt", 'w')   # this will print when ever overlap detected between a
            f.write( input_file + '\n')
        #f.close()


###############################################################################
# Extracting rows of EKG by using connected component algorithm

def extract_connected_components(image):
#    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    # Tricky part 
    # Mac and Linux drivers work differently
    # Hence for image thresholding we need to check 
	
    ret, thresh = cv2.threshold(image,80,150,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU) # for LINUX
    #ret, thresh = cv2.threshold(image,80,150,cv2.THRESH_BINARY_INV) # for LINUX

    #cv2.imshow("image",thresh)

    # find contours in the thresholded image
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
    	cv2.CHAIN_APPROX_SIMPLE)[-2]

#    print("[INFO] {} unique contours found".format(len(cnts)))

#    Initialize an image and make the content equal to 255
    extracted_wave_image=255-(image*0)

    # loop over the contours
    for (i, c) in enumerate(cnts):
    	# draw the contour
        #print len(c)
        # This will work for both leads and rythm strip
    	if len(c)>400 :
        	((x, y), _) = cv2.minEnclosingCircle(c)
                #print len(c)
                #cv2.putText(image, "#{}".format(i + 1), (int(x) - 10, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                cv2.drawContours(extracted_wave_image, [c], -1, (0, 0, 0), 2)
    #cv2.imshow("Connected components",extracted_wave_image)
    #extracted_image="extracted_" + file.split('.')[0] +".jpg"
    #cv2.imwrite(extracted_image,extracted_wave_image)
    #cv2.waitKey(0)
    return extracted_wave_image


###############################################################################
########################  Extract text from PDF ###############################

# For extracting the header contents from the PDF

def extract_EKG_Header(input_file):

    #################### read input text ###################
    os.chdir(os.path.expanduser(input_folder))
#    print(input_file)
    #os.path.isfile(input_file)
    os.system(''.join("pdftotext " + input_file + " -layout"))

    #################### Extract from text #################

    Vent_Rate = " "
    PR_Interval = " "
    QRS_Duration = " "
    QT_QTc = " "
    PRT = "$"
    BP = " "
    SCALE_X = " "
    SCALE_Y = " "
    SIGNAL = " "
    ID = " : "
    Name = "$"
    Date = "$"
    Sex = " "
    DOB = "$"
    ACCOUNT = "ACCOUNT$:"
    Room = " "
    Loc = " "
    Referredby = " "
    Confirmedby = " "
    Notes_joined= " "


    try:
        file_header = open(input_for_extracting_JSON, 'r').readlines()
    except Exception as e:
        print(e)

    for i in file_header:
        try:
            thisline = list(i)
            for j in range(1, len(i) - 2):
                first = i[j]
                second = i[j + 1]
                third = i[j + 2]
                if first != ' ' and second == ' ' and third != ' ':
                    thisline[j + 1] = '$'
            str1 = ''.join(thisline)
#        print str1 ############################
            words = str1.split()
#        print len(words)
        except Exception as e:
            print(" Error in parsing the txt file ")
            print(e)
            print(str1)

# Before For loop of words

        try:
            Notes = " "
            for i in range(0, len(words)):
                #print words[i]
                if "Vent" in words[i] and words[i].startswith("Vent"):
                    Vent_Rate = words[i + 1]
                if "interval" in words[i] and words[i].startswith("PR"):
                    PR_Interval = words[i + 1]
                if "duration" in words[i] and words[i].startswith("QRS"):
                    QRS_Duration = words[i + 1]
                if "QT/QTc" in words[i]:
                    QT_QTc = words[i + 1]
                if "P-R-T" in words[i]:
#                   PRT = words[i + 1]
                    PRT = words[i + 1].replace('$', ';') + ';' + (words[i + 2])
#                   print words[i+2]
                if "BP" in words[i] and "BPM" not in words[i]:
                    BP = words[i + 1]
                if "mm/s" in words[i]:
                    SCALE_X = words[i]
                    global X_scale_obtained_from_PDF
                    X_scale_obtained_from_PDF =int(re.sub('[^0-9]', '', SCALE_X))
                    global X_scale
                    X_scale = (X_scale_obtained_from_PDF *12.0 )
                if "mm/mV" in words[i]:
                    SCALE_Y = words[i]
                if "Hz" in words[i]:
                    SIGNAL = words[i]
                #if "ID" in words[i] and "CID"not in words[i] and "EID"not in words[i]:
                #if "ID" in words[i]:
                if words[i].startswith("ID") and ":" in words[i] :
                    #print((words[i]))
                    ID = words[i]
                    Name = words[i - 1]
                    Date = words[i + 1]
                if "Male" in words[i]:
                    Sex = words[i]
                if "Female" in words[i]:
                    Sex = words[i]
                if "yr" in words[i]:
                    DOB = words[i]
                if "ACCOUNT" in words[i]:
                    ACCOUNT = words[i]
                if "Room" in words[i] and ":" in words[i]:
                    Room = (words[i])
                if "Loc" in words[i] and ":" in words[i]:
                    Loc = (words[i])
                if "Confirmed" in words[i] and ":" in words[i]:
                    Confirmedby = (words[i])
                if "Referred" in words[i]and ":" in words[i]:
                    Referredby = words[i]
                if "Technician" in words[i]:
                    Technician = words[i]
                alphabet= ('*','A' ,'B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z')
                if (len(words[i]) >15 and "ACCOUNT" not in words[i] and "EID" not in words[i]
                and "BRIGHAM" not in words[i] and "EDT" not in words[i] and "ORDER" not in words[i]
                and words[i].startswith(alphabet)
                and "Technician" not in words[i] and "Confirmed" not in words[i]and "Referred" not in words[i]
                and "technician" not in words[i] and "confirmed" not in words[i]and "referred" not in words[i] and "vent" not in words[i]
                and "Vent" not in words[i] and "Test ind " not in words[i]
                and Name.split(',')[0] not in words[i]
                or "ECG" in words[i]) :
#                    print words[i].replace('$',' ')
                    global Notes_joined
                    Notes_joined = Notes_joined + words[i].replace('$', ' ') + '. '

        except Exception as e:
                print "Error in finding the key terms in the txt file"
                print e

    try:

        PRT= PRT.replace('$', ' ')
        ID = ID.split(':')[1]
        Name = Name.replace('$',' ')
        Date = Date.replace('$',' ')
        DOB = DOB.replace('$', ' ')
        if len(ACCOUNT)>10:
            ACCOUNT=ACCOUNT.replace('$', ' ').split('ACCOUNT')[1].replace(':','')
            #ACCOUNT = int(filter(str.isdigit, ACCOUNT.split('ACCOUNT:')[1]))
        else :
            ACCOUNT= " "
        if len(Loc)>4 :
            Loc = Loc.replace('$', ' ').split(':')[1]
        else :
            Loc =" "
        if len(Confirmedby)>15 :
            Confirmedby = Confirmedby.replace('$', ' ').split(':')[1]
        else :
            Confirmedby = " "

        if len(Referredby) > 10 :
            Referredby = Referredby.replace('$', ' ').split(':')[1]
        else :
            Referredby = " "
        if len(Room)>5 :
            Room = Room.replace('$', ' ').split(':')[1]
        else :
            Room =" "
    except Exception as e:
        print e

    data = {"Vent_Rate" : Vent_Rate, "PR_Interval" : PR_Interval, "QRS_Duration" : QRS_Duration, "QT_QTc":QT_QTc, "PRT":PRT, "BP":BP, "SCALE_X":SCALE_X,
        "SCALE_Y":SCALE_Y, "SIGNAL":SIGNAL, "ID":ID, "Name":Name, "Date":Date, "Sex":Sex, "DOB": DOB, "ACCOUNT":ACCOUNT,
        "Room":Room , "Loc": Loc ,"Confirmed By ":Confirmedby ,"Referred By":Referredby, "Notes" : Notes_joined}  # lint:ok

    ID_as_filename = ''.join(str(ID))

    append_name = input_file.split('.')[0].replace('/', '_')
    ID_as_filename = join(ID_as_filename + '_' + append_name)

    file_json = ''.join(ID_as_filename + ".json")
    file_json = os.path.join(output_folder, file_json)

    with io.open(file_json, 'w', encoding='utf-8') as f:
        f.write(unicode(json.dumps(data, ensure_ascii=False)))
    return ID_as_filename
    #print(" PDF reading and extraction completed")

###############################################################################

# Function to fetch CSV and header ,
# from here all other functions are called
# main among the all the functions

def getSignalCSVandJSONfromPdf():
#    print "getSignalCSVandJSONfromPdf"
    try:
        ID_as_filename = extract_EKG_Header(input_file)
    except Exception as e:
        logger.error("Problem in Extracting header : %s " % (input_file))
        logger.error(e)
    ################################################################

    # If image is in predefined dimension
    if img.shape[0] == EXPECTED_Y_LENGTH and img.shape[1] == EXPECTED_X_LENGTH:
        global df_merged
        df_merged = pd.DataFrame(np.nan, index=list(range(0, 0)), columns=[])

        base = [None] * (GRID_END_Y_AXIS)
        index_top_down = []
        global index_gap_between_signals
        index_gap_between_signals = [600] # need to hard code because some letters can come in btw..

        global overlap_flag
        overlap_flag = 0

        for x in range(GRID_BEGIN_Y_AXIS, GRID_END_Y_AXIS):
            for y in range(1, SEARCH_STARTING_POINT_SIGNAL):
                if img[x, y] == 0:
                    base[x] = y
                    index_top_down.append(x)
#                    print index_top_down
                    break
#        print base
        # this has the number of signal waves presnet in the grid
        test=[]
#        print len(index_top_down)
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
#        print((index_top_down))

    ###############################################################
        try:
            # To find cut between signals or finding overlap .
            # Done by raster scaning and adding
            # This part is tricky , because we have to identify if it overlaps or not
            # ie.. distinguish between case 4 and case 5 overlap

            find_the_cut_between_signals()

        except Exception as e:
            logger.error("Error in function to find overlap and cut between signals in %s" % (input_file))
            logger.error(e)
    ###############################################################

        # With new algorithm we should get inside the loop in all cases
        # As even if we have overlap we can get the signals ,
        # but overlap regions has to marked as
        if overlap_flag == 0 or overlap_flag == 1:  # 0
            margin = 20
            index_final_split = [index_gap_between_signals[0], index_top_down[1]-margin,
            index_top_down[0]+margin,index_top_down[2]-margin, index_top_down[1]+margin,
            int(index_gap_between_signals[3])]
            #print index_top_down
            #print index_gap_between_signals
            #print index_final_split
            iter_rows = 3  # number of rows to be extracted
            c=-1
            for i in range(0, iter_rows*2,2):
                c= c+1
                E, I = extract_one_row_automatic(index_final_split[i],index_final_split[i+1])  # HI
                for j_index in range(0, 4):
                    plot_splitted_EKG(c, j_index, j_index + 1, I, E)
#                    print((str(i) + str(j_index)))
#            print((ID_as_filename))
    #        df_merged.to_csv(ID_as_filename + ".csv")  # Save as CSV
            logger.error(len(df_merged))
            df_merged.to_csv(os.path.join(output_folder, (ID_as_filename + ".csv")), index=False)  # Save as CSV
            os.remove(input_for_extracting_JSON)
            os.remove(input_image_from_pdf)
        else:
            logger.error("Signal Overlap : %s " % (input_file))

###############################################################################
#                                main
###############################################################################


#####################  SET Variables ##########################################

global input_folder,output_folder,input_pdf,input_file,input_for_extracting_JSON,temp_folder,input_image_from_pdf
global EXPECTED_X_LENGTH,EXPECTED_Y_LENGTH,Y_scale,X_scale,GRID_BEGIN_Y_AXIS,GRID_END_Y_AXIS,SEARCH_STARTING_POINT_SIGNAL
global SIGNAL_UPPER_LIMIT_FROM_CENTER,SIGNAL_LOWER_LIMIT_FROM_CENTER
global index_top_down
global lead_names
global index_gap_between_signals

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
GRID_BEGIN_Y_AXIS = 650 # 650 #700  # To find the starting line of each signal
GRID_END_Y_AXIS =  2410 #EXPECTED_Y_LENGTH # To find the bottom part of the siganl waves .Not to keep 2550.
SEARCH_STARTING_POINT_SIGNAL = 65 #55 # search in x-axis till 65
    # 170 255
SIGNAL_UPPER_LIMIT_FROM_CENTER = 120     # if 4 waves present
SIGNAL_LOWER_LIMIT_FROM_CENTER = 120     # if 4 waves present

index_top_down = []

lead_names = [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"],
             ["III", "aVF", "V3", "V6"]]

count_total_files = 0
count_total_processed_files = 0
count_overlap_files = 0
###############################################################################

input_folder = sys.argv[1]
output_folder = sys.argv[2]

###############################################################################
logger = logging.getLogger('MAIN')
hdlr = logging.FileHandler(os.path.join('EKG_EXTRACTION.log'))
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.WARNING)
###############################################################################

if (os.path.isdir(output_folder)):
    # Check if input directory exists
    if (os.path.isdir(input_folder)):
        onlyfiles = [f for f in listdir(input_folder) if isfile(join(input_folder, f)) if f.endswith('.pdf')]
        # Display files in the input folder
        print("The PDF files present :")
        print((onlyfiles))

        # Check if the input folder is empty : proceed only if not empty
        if len(onlyfiles) > 0:
            for file in onlyfiles:
                input_pdf = os.path.join(input_folder, file)
                print("#############################################")
    #            print(input_pdf)
                input_file = file
                print(input_file)
                input_for_extracting_JSON = os.path.join(input_folder,file.replace('.pdf', '.txt'))
    #            print(input_for_extracting_JSON)
                temp_folder = output_folder
    #            print(temp_folder)
                input_image_from_pdf = os.path.join(temp_folder, file.replace('.pdf', '.jpg'))
    #            print(input_image_from_pdf)
                ###################################################################
                try:
                    Input_as_pdf = PyPDF2.PdfFileReader(input_pdf)
                    pdf_x_length = Input_as_pdf.getPage(0).mediaBox[3]
                    pdf_y_length = Input_as_pdf.getPage(0).mediaBox[2]
                except Exception as e:
                    logger.error("Error in PdfFileReader in %s" % (input_file))
                    logger.error(e)
#                print pdf_x_length
#                print pdf_y_length
#                print Input_as_pdf.getNumPages()
                # check if the dimension of pdf is as expected
                #
                if PDF_DIM_X == pdf_x_length and PDF_DIM_Y == pdf_y_length and Input_as_pdf.getNumPages()==1:
                    #  Converting input pdf to image (jpg with dpecified dpi) #########
                    print("Converting PDF to image (jpg)")
                    print input_pdf
                    img = PythonMagick.Image()
                    img.density("300")
                    img.read(input_pdf)  # read in at 300 dpi
                    img.write(input_image_from_pdf)
                    img = cv2.imread(input_image_from_pdf, 0)
    #                print((img.shape[0]))
    #                print((img.shape[1]))
                    #cv2.imshow('Converted image', img)
    #                print((img.shape))
                    ###################################################################
                    try:
                        getSignalCSVandJSONfromPdf()
                    except Exception as e:
                        print " Problem with input file. Either it's not an EKG file or has Overlapping "
                        logger.error(" Problem with input file. %s is not a proper EKG file." % (input_file) )
                        logger.error(e)
#                        os.remove(input_for_extracting_JSON)
#                        os.remove(input_image_from_pdf)

                elif PDF_DIM_X == pdf_x_length and PDF_DIM_Y == pdf_y_length and Input_as_pdf.getNumPages() != 1:  # Dimension error
                        print " Problem with file. It's not an EKG file."
                        logger.error(" Problem with file. %s is not a proper EKG file." % (input_file) )
#                        os.remove(input_for_extracting_JSON)
                        os.remove(input_image_from_pdf)
                else:
                    print("PDF dimension mismatch.The dimension of %s is [%d X %d] instead of [%d X %d]." % (
                        input_file, pdf_x_length , pdf_y_length, PDF_DIM_X, PDF_DIM_Y))
                    logger.error("PDF dimension mismatch.The dimension of %s is [%d X %d] instead of [%d X %d]." % (
                        input_file, pdf_x_length , pdf_y_length, PDF_DIM_X, PDF_DIM_Y))

        else:  # Check if the input folder is empty : proceed only if not empty
            print("Input directory Empty ! Please check your input folder")
            logger.error("Input directory Empty ! Please check your input folder")

    else:  # Check if input directory exists
        print("NO such directory ! Please check your Input path/folder")
        logger.error("NO such directory ! Please check your Input path/ Input folder")

else:
    print("NO such directory ! Please check your Output path/ Output folder")
    logger.error("NO such directory ! Please check your Output path/ Output folder")

###############################################################################
#                         END OF MAIN
###############################################################################

