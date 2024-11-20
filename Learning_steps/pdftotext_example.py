#################### PDF to text conversion ############

import os
import re
import json
from json import dumps, load

newdir="input"
os.chdir(os.path.expanduser(newdir))
os.path.isfile("sample4.pdf")
os.system("pdftotext sample4.pdf -layout")


#################### Extract from text #################

file = open('sample4.txt', 'r').readlines()

for i in file:
    thisline=list(i)
    for j in range(1,len(i)-2):
        first= i[j]
        second=i[j+1]
        third=i[j+2]
        if first !=' ' and second == ' '  and third != ' ' :
            thisline[j+1]='$'
    str1 = ''.join(thisline)

#    print str1 ############################
    words=str1.split();


    for i in range(0,len(words)):
        # print the word
        if "Ve" in words[i] :
            Vent_Rate= words[i+1]
        if "PR" in words[i] :
            PR_Interval=words[i+1]
        if "QR" in words[i] :
            QRS_Duration= words[i+1]
        if "QT/" in words[i] :
            QT_QTc=words[i+1]
        if "P-R-T" in words[i] :
            PRT= words[i+1]
#            print words[i+2]
        if "BP" in words[i] and "BPM" not in words[i]:
            BP = words[i+1]
        if "mm/s" in words[i] :
            SCALE_X =words[i]
        if "mm/mV" in words[i] :
            SCALE_Y=words[i]
        if "Hz" in words[i] :
            SIGNAL= words[i]
        if "ID" in words[i] and "CID"not in words[i] and "EID"not in words[i]:
            print words[i]
            ID= words[i]
            Name= words[i-1]
            Date= words[i+1]
        if "Male" in words[i] :
            Sex= words[i]
        if "Female" in words[i] :
            Sex= words[i]
        if "yr" in words[i]:
            DOB=     words[i]
        if "ACC" in words[i]:
            ACCOUNT= words[i]


        if "Room" in words[i] :
            print words[i]
        if "Loc" in words[i] :
            print words[i]
        if "Confirmed" in words[i]:
            print words[i]
        if "Referred" in words[i]:
            print words[i]
        if "Tech" in words[i] :
            print words[i]

data={"Vent_Rate":Vent_Rate,"PR_Interval":PR_Interval,"QRS_Duration":QRS_Duration,"QT_QTc":QT_QTc,"PRT":PRT.replace('$',' '),"BP":BP,"SCALE_X":SCALE_X,
    "SCALE_Y":SCALE_Y,"SIGNAL":SIGNAL,"ID":ID.split(':')[1],"Name":Name.replace('$',' '),"Date":Date.replace('$',' '),"Sex":Sex,"DOB":DOB.replace('$',' '),"ACCOUNT":ACCOUNT.replace('$',' ').replace('$',' ').split(':')[1]}  # lint:ok


file=''.join(str(ID.split(':')[1])+".txt")
import io, json
with io.open(file, 'w', encoding='utf-8') as f:
    f.write(unicode(json.dumps(data, ensure_ascii=False)))