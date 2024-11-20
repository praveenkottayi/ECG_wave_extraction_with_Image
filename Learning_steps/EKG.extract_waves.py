
import numpy as np
import cv2
import pandas as pd
import array
import matplotlib.pyplot as plt
import PyPDF2
import PythonMagick

##########################################################################
def extract_one_row(i):
    x1=index_top_down[i]
    x2=x1
    y1=0
    y2=3300
    img1=img[x1-SIGNAL_UPPER_LIMIT_FROM_CENTER:x2+SIGNAL_LOWER_LIMIT_FROM_CENTER,y1:y2]
#    cv2.imshow(str(i),img1)
    thresh1=image_enhancement(img1) # function call for thresholding
    print "size of thrshld" + str(thresh1.shape)
    ecg_wave=digitize_full_row_ecg(thresh1)
    index_split=split_EKG_into_leads(ecg_wave)
    ecg_wave_normalized=normalize_EKG(ecg_wave)
    cv2.waitKey(0)
    return (ecg_wave_normalized ,index_split)


##########################################################################
def extract_one_row_automatic(i):
    x1=index_top_down[i]
    x2=x1
    y1=0
    y2=3300
    img1=img[x1-SIGNAL_UPPER_LIMIT_FROM_CENTER:x2+SIGNAL_LOWER_LIMIT_FROM_CENTER,y1:y2]
#    cv2.imshow(str(i),img1)
    thresh1=image_enhancement(img1) # function call for thresholding
    print "size of thrshld" + str(thresh1.shape)
    ecg_wave=digitize_full_row_ecg(thresh1)
    index_split=split_EKG_into_leads(ecg_wave)
    ecg_wave_normalized=normalize_EKG(ecg_wave)
    cv2.waitKey(0)
    return (ecg_wave_normalized ,index_split)


##########################################################################


def image_enhancement(img1):
    ret,thresh1 = cv2.threshold(img1,175,255,cv2.THRESH_BINARY)
    cv2.imshow('EKG_processed_wave', thresh1)
    return thresh1

########################  Digitize wave signal  ##########################

def digitize_full_row_ecg(thresh):
    xxx = [None]*thresh.shape[1]
    ecg_wave=[None]*thresh.shape[1]
    X_range=thresh.shape[0]
    for y in range(1,thresh.shape[1]):
        for x in range(1,X_range):
            if thresh[x,y]==0:
                ecg_wave[y] = X_range-x
                break
    return ecg_wave

def split_EKG_into_leads(ecg_wave):
    index_break_wave=[item for item in range(len(ecg_wave)) if ecg_wave[item] == None]
#    print index_break_wave
    index_L_R=[]
    for item in range(1,len(index_break_wave)-1):
        difference= index_break_wave[item]-index_break_wave[item+1]
        if (-1*difference>100):
            start_of_wave=index_break_wave[item]
            index_L_R.append(start_of_wave)
    index_L_R.append(index_break_wave[len(index_break_wave)-1])
    return index_L_R


def normalize_EKG(ecg_wave):
    ecg_wave_normalized=ecg_wave
    for item in range(index_top_down[0],index_top_down[1]):
        if (ecg_wave[item]>10):
            norm_value=ecg_wave[item]
            break
    print norm_value
    for item in range(1,len(ecg_wave)):
        if ecg_wave[item]!= None:
            ecg_wave_normalized[item]=ecg_wave[item]-norm_value
    return ecg_wave_normalized

##########################################################################

def plot_splitted_EKG(i,a,b,index,ecg_wave_normalized):
    df = pd.DataFrame(np.nan, index=range(index[a],index[b]), columns=[])

    df['Lead']="Lead"+" "+lead_names[i][a]
#    df['i']=i
#    df['j']=a
    df['absoluteY']=ecg_wave_normalized[index[a]:index[b]]
    df['actual_X']=[float(x /X_scale) for x in range(index[a],index[b]) ]
    df['actual_Y']=[float(x /Y_scale) for x in df['absoluteY'] ]

    global df_merged
    df_merged =df_merged.append(df)
#    df.to_csv('output/df_ '+str(i)+'.csv') # Save as CSV

    plt.plot(ecg_wave_normalized[index[a]:index[b]])
    plt.title('Extracted wave '+lead_names[i][a])
    plt.xlabel('time')
    plt.ylabel('mV')
    plt.show()


##########################################################################
#                       main
##########################################################################

#####################  SET  input  ############################

input_folder="input/"
file_name="Brigham_Sample_EKG_full"
input_pdf=input_folder+file_name+".pdf"
output_folder="output/"


temp_folder="temp"
input_image_from_pdf=temp_folder+file_name+".jpg"

# check DIR exists
# check file exists

# check .. dimension
# remove temp images
#####################  Read INPUT pdf  ########################

print " Reading PDF "
img = PythonMagick.Image()
img.density("300")
img.read(input_pdf)        #  read in at 300 dpi


img.write(input_image_from_pdf)

img = cv2.imread(input_image_from_pdf,0)
print img.shape[0]
print img.shape[1]
#cv2.imshow('Converted image', img)
print img.shape

################################################################

df_merged = pd.DataFrame(np.nan, index=range(0,0), columns=[])

lead_names= [["I","aVR","V1","V4"],["II","aVL","V2","V5"],["III","aVF","V3","V6"]]

Y_scale=118.0
X_scale=(60.0*5)

GRID_BEGIN_Y_AXIS=700
GRID_END_Y_AXIS=img.shape[0]
SEARCH_STARTING_POINT_SIGNAL=55

SIGNAL_UPPER_LIMIT_FROM_CENTER=170     # if 4 waves present
SIGNAL_LOWER_LIMIT_FROM_CENTER=255     # if 4 waves present

#SIGNAL_UPPER_LIMIT_FROM_CENTER=250     # if 3 waves present
#SIGNAL_LOWER_LIMIT_FROM_CENTER=350     # if 3 waves present

base=[None]*(GRID_END_Y_AXIS)

index_top_down=[]

for x in range(GRID_BEGIN_Y_AXIS,GRID_END_Y_AXIS):
    for y in range(1,SEARCH_STARTING_POINT_SIGNAL):
        if img[x,y]==0:
            base[x]= y;
            index_top_down.append(x);
            break
#print base
print index_top_down # this has the number of signal waves presnet in the grid

iter= 3 # number of rows to be extracted
for i in range(0,iter):
    print index_top_down
    E ,I= extract_one_row(i)  #

    for j_index in range(0,4):
        plot_splitted_EKG(i,j_index,j_index+1,I,E)
        print str(i)+str(j_index)

df_merged.to_csv(output_folder+file_name+'.csv') # Save as CSV

################################################################



