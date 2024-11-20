# -*- coding: utf-8 -*-
import os
from os import listdir
from os.path import isfile, join
import sys

inFile = sys.argv[1]
outFile = sys.argv[2]

onlyfiles = [f for f in listdir(inFile) if isfile(join(inFile, f)) if f.endswith('.pdf')]
print onlyfiles


def manipulateData(lines):
    lines = 2 * lines
    return lines


def output(file):
    with open((os.path.join(outFile, str(file))), 'w') as o:
        for line in processedLines:
            o.write(line)

for file in onlyfiles:
#    print(os.path.join(inFile, file))
    with open(os.path.join(inFile, file), 'r') as i:
        lines = i.readlines()
        processedLines = manipulateData(lines)
        output(file)

