# ECG_wave_extraction_with_Image
 To extract Headers and signals from the scan images of ECG. This signals can be used further as features for the ML/AI models.
 
# Context 
EKG files are a core source of information on identifying heart related conditions.  Most feature extraction mechanisms focus on retrieving core information in an efficient form.  Medical practitioners have pre identified health related features (distance between key parts of the waves, eg. QRS complex) and technologists have focused on efficient extraction based on downsampling the data through mathematical representations.  We instead wish to reexamine the area by extracting the lowest level features of the EKG and use those as inputs into ML / NN techniques to extract features or make predictions.

# Goal
Develop a modular software approach to extract low level features from EKG files (images / PDFs).  The software should be able to operate on a single file or directories of files in an automated way.  The output of the execution should be a csv or JSON file that can be imported easily into other environments.  The exact format of the output should be pluggable to support later changes (ie varying input images that may have more or less data in them)

# Assumptions
software must be easily deployable without any external dependencies (non cloud based).
no dependency on commercially licensed software
use a relevant mainstream programming languages (Java, C/C++, Python, R)
the program can be pointed at a file(s) or at a director (and will process every file in that directory), output is written to wherever the program is run from or configurable directory.

