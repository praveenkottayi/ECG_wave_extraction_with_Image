###############################################################
#         EKG PROJECT
#         Extarct Text and Signal wave from an EKG PDF
#         Save the signal into CSV or DB
###############################################################

import numpy as np
import pandas as pd
import array
import matplotlib.pyplot as plt
import cv2
import PyPDF2
import PythonMagick

#####################  SET  input files / DIR #################


input_pdf="Brigham_Sample_EKG_full.pdf"
input_image_from_pdf="Brigham_Sample_converted_image.jpg"


#####################  Read INPUT pdf  ########################

print " Reading PDF "
img = PythonMagick.Image()
img.density("300")
img.read(input_pdf) # read in at 300 dpi


img.write(input_image_from_pdf)

img = cv2.imread(input_image_from_pdf,0)
print img.shape[0]
print img.shape[1]
cv2.imshow('Converted image', img)

########################  Define the boundary of signals  ####

signal_number=0
x1=1000
x2=1400
X_range=x2-x1
x1=1100+(X_range*signal_number)
x2=1500+(X_range*signal_number)

y1=0
y2=3300


########################  Extract first wave signal  #########


img1=img[x1:x2,y1:y2]
print img1.shape
ret,thresh1 = cv2.threshold(img1,175,255,cv2.THRESH_BINARY)
#ret,thresh1 = cv2.adaptiveThreshold(img1,170,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C)

cv2.imshow('EKG_original_wave', img1)
cv2.waitKey(0)

xxx = [None]*thresh1.shape[1]
ecg_wave=[None]*thresh1.shape[1]

########################  Digitize first wave signal  #######

for y in range(1,y2):
    for x in range(1,X_range):
        if thresh1[x,y]==0:
            ecg_wave[y] = X_range-x
            break

########################  Extract text from PDF #############

print (" PDF reading in progress")
pdfFileObj = open(input_pdf, 'rb')

pdfReader = PyPDF2.PdfFileReader(pdfFileObj)
pageObj = pdfReader.getPage(0)

print ( pdfReader.numPages )
print (pageObj.extractText().strip().split())
Header=pageObj.extractText().strip().split()
print(Header[0])

print "title = %s" % (pdfReader.getDocumentInfo().title)

print " PDF reading in completed"

####################  Display digitized graph  ##############

cv2.imshow('EKG_THRESHOLDED', thresh1)

k = cv2.waitKey(0)


###################       split the EKG wave  ##################################

print ecg_wave

print "index"

print ecg_wave.index(None)
index_break_wave=[item for item in range(len(ecg_wave)) if ecg_wave[item] == None]
print index_break_wave

index=[]
for item in range(1,len(index_break_wave)-1):
    difference= index_break_wave[item]-index_break_wave[item+1]
    if (-1*difference>100):
        start_of_wave=index_break_wave[item]
        index.append(start_of_wave)
index.append(index_break_wave[len(index_break_wave)-1])
print index

#######################  Normalize and display the waves #######################


ecg_wave_normalized=ecg_wave
for item in range(index[0],index[1]):
    if (ecg_wave[item]>10):
        norm_value=ecg_wave[item]
        break

print norm_value

for item in range(1,len(ecg_wave)):
    if ecg_wave[item]!= None:
        ecg_wave_normalized[item]=ecg_wave[item]-norm_value


plt.plot(ecg_wave_normalized)
plt.title('Extracted wave')
plt.xlabel('time')
plt.ylabel('mV')
plt.show()


ecg=pd.DataFrame(np.nan, index=range(0,0), columns=[])
ecg= pd.DataFrame(ecg_wave_normalized)

print type(ecg_wave_normalized)
print type(ecg)
print "ecg............................................."

print ecg

plt.plot([i for i in ecg])
plt.title('Scaled wave')
plt.xlabel('time')
plt.ylabel('mV')
plt.show()

#plt.plot(ecg_wave_normalized[index[0]:index[1]])
#plt.title('Extracted wave 1')
#plt.xlabel('time')
#plt.ylabel('mV')
#plt.show()

#plt.plot(ecg_wave_normalized[index[1]:index[2]])
#plt.title('Extracted wave 2')
#plt.xlabel('time')
#plt.ylabel('mV')
#plt.show()

#plt.plot(ecg_wave_normalized[index[2]:index[3]])
#plt.title('Extracted wave 3')
#plt.xlabel('time')
#plt.ylabel('mV')
#plt.show()

#plt.plot(ecg_wave_normalized[index[3]:index[4]])
#plt.title('Extracted wave 4')
#plt.xlabel('time')
#plt.ylabel('mV')
#plt.show()

#############################################################

if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.destroyAllWindows()

####################  END  ##################################












