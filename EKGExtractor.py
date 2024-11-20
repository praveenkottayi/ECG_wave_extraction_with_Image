import logging
import numpy as np
import cv2
import pandas as pd
#import array
import matplotlib.pyplot as plt
import PyPDF2
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
# Extacting one row
#
def extract_one_row_automatic(i):
#    print "extract_one_row_automatic"
    global index_gap_between_signals
    y1 = index_gap_between_signals[i]
#    print ("...................automatic................")
#    print (i)
#    print (x1)
    y2 = index_gap_between_signals[i + 1]
#    print (x2)
    x1 = 0
    x2 = EXPECTED_X_LENGTH
    img1 = img[y1:y2, x1:x2]

#    cv2.imshow(str(i),img1)
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


def image_enhancement(img1):
#    print "image_enhancement"
    ret, thresh1 = cv2.threshold(img1, 175, 255, cv2.THRESH_BINARY)
#    cv2.imshow('EKG_processed_wave', thresh1)
    return thresh1

########################  Digitize wave signal  ##########################


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
    df_merged = df_merged.append(df) #  No problem with the code
#    logger.error("plot splitted EKG")
#    logger.error(len(df_merged))
#    df.to_csv('df_ ' + str(i) + '.csv')     # Save as CSV

#    plt.plot(ecg_wave_normalized[index[a]:index[b]])
#    plt.title('Extracted wave ' + lead_names[i][a])
#    plt.xlabel('time')
#    plt.ylabel('mV')
#    plt.show()


def find_the_cut_between_signals():
#    print "find_the_cut_between_signals"
    thresh1 = image_enhancement(img)  # function call for thresholding
    sum_row_to_find_cut = np.sum(thresh1, axis=1)
    sum_row_to_find_cut = (EXPECTED_X_LENGTH * 255) - sum_row_to_find_cut
#    sum_part = sum_row_to_find_cut[700:index_top_down[4]]
#    plt.plot(sum_part)
#    plt.show()
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
        index_gap_between_signals.append(np.ceil(np.mean(index_gap)))


########################  Extract text from PDF #############

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
                PRT = words[i + 1]
    #            print words[i+2]
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
#    print(".................................................")
#    print(ID_as_filename)
#    print(".................................................")
    file_json = ''.join(ID_as_filename + ".json")
    file_json = os.path.join(output_folder, file_json)
    with io.open(file_json, 'w', encoding='utf-8') as f:
        f.write(unicode(json.dumps(data, ensure_ascii=False)))
    return ID_as_filename
    print(" PDF reading in completed")


def getSignalCSVandJSONfromPdf():
#    print "getSignalCSVandJSONfromPdf"

    ID_as_filename = extract_EKG_Header(input_file)
    ################################################################

    # If image is in predefined dimension
    if img.shape[0] == EXPECTED_Y_LENGTH and img.shape[1] == EXPECTED_X_LENGTH:
        global df_merged
        df_merged = pd.DataFrame(np.nan, index=list(range(0, 0)), columns=[])

#        lead_names = [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"],
#             ["III", "aVF", "V3", "V6"]]
        base = [None] * (GRID_END_Y_AXIS)
#        index_top_down = []
        global index_gap_between_signals
        index_gap_between_signals = [GRID_BEGIN_Y_AXIS]
        global overlap_flag
        overlap_flag = 0

        for x in range(GRID_BEGIN_Y_AXIS, GRID_END_Y_AXIS):
            for y in range(1, SEARCH_STARTING_POINT_SIGNAL):
                if img[x, y] == 0:
                    base[x] = y
                    index_top_down.append(x)
                    break
        #print base
        # this has the number of signal waves presnet in the grid
#        print((index_top_down))
    ###############################################################

        find_the_cut_between_signals()
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




##########################################################################
#                       main
##########################################################################


#####################  SET  input  #######################################

global input_folder,output_folder,input_pdf,input_file,input_for_extracting_JSON,temp_folder,input_image_from_pdf
global EXPECTED_X_LENGTH,EXPECTED_Y_LENGTH,Y_scale,X_scale,GRID_BEGIN_Y_AXIS,GRID_END_Y_AXIS,SEARCH_STARTING_POINT_SIGNAL,SIGNAL_UPPER_LIMIT_FROM_CENTER,SIGNAL_LOWER_LIMIT_FROM_CENTER
global index_top_down
global lead_names
#global overlap_flag
#global df_merged
global index_gap_between_signals
#df_merged = pd.DataFrame(np.nan, index=list(range(0, 0)), columns=[])

#print "main"
EXPECTED_X_LENGTH = 3300
EXPECTED_Y_LENGTH = 2550
PDF_DIM_X = 792
PDF_DIM_Y = 612

Y_scale = 118.0 #    may not be significant as this will be normalized later on
global X_scale_obtained_from_PDF
X_scale_obtained_from_PDF = 1
X_scale = (25 * 12.0)
GRID_BEGIN_Y_AXIS = 700
GRID_END_Y_AXIS = EXPECTED_Y_LENGTH
SEARCH_STARTING_POINT_SIGNAL = 55
    # 170 255
SIGNAL_UPPER_LIMIT_FROM_CENTER = 120     # if 4 waves present
SIGNAL_LOWER_LIMIT_FROM_CENTER = 120     # if 4 waves present

index_top_down = []
#df_merged = []

lead_names = [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"],
             ["III", "aVF", "V3", "V6"]]
##################################################################

input_folder = sys.argv[1]
output_folder = sys.argv[2]


logger = logging.getLogger('MAIN')
hdlr = logging.FileHandler(os.path.join('EKG_EXTRACTION.log'))
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.WARNING)

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
                Input_as_pdf = PyPDF2.PdfFileReader(input_pdf)
                pdf_x_length = Input_as_pdf.getPage(0).mediaBox[3]
                pdf_y_length = Input_as_pdf.getPage(0).mediaBox[2]
                print pdf_x_length
                print pdf_y_length
                print Input_as_pdf.getNumPages()
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
                    except:
                        print " Problem with file. Guess its not an EKG file."
                        logger.error(" Problem with file. Guess %s not a proper EKG file." % (input_file) )
                        os.remove(input_for_extracting_JSON)
                        os.remove(input_image_from_pdf)

                elif PDF_DIM_X == pdf_x_length and PDF_DIM_Y == pdf_y_length and Input_as_pdf.getNumPages() != 1:  # Dimension error
                        print " Problem with file. Guess its not an EKG file."
                        logger.error(" Problem with file. Guess %s not a proper EKG file." % (input_file) )
                        os.remove(input_for_extracting_JSON)
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

    #######################################################################
    #                         END OF MAIN
    #######################################################################

