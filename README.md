# ECG_wave_extraction_with_Image
 This component will help in digitalizing old ECG records.To extract Headers and signals from the scan images of ECG. This signals can be used further as features for the ML/AI models.
 If its a direct pdf generated from the machine, then the content could be read using pdf extractors easily. 
 
# Context 
EKG files are a core source of information on identifying heart related conditions.  Most feature extraction mechanisms focus on retrieving core information in an efficient form.  Medical practitioners have pre identified health related features (distance between key parts of the waves, eg. QRS complex) and technologists have focused on efficient extraction based on downsampling the data through mathematical representations.  We instead wish to reexamine the area by extracting the lowest level features of the EKG and use those as inputs into ML / NN techniques to extract features or make predictions.

# Goal
Develop a modular software approach to extract low level features from EKG files (images / PDFs).  The software should be able to operate on a single file or directories of files in an automated way.  The output of the execution should be a csv or JSON file that can be imported easily into other environments.  The exact format of the output should be pluggable to support later changes (ie varying input images that may have more or less data in them)

# Assumptions
software must be easily deployable without any external dependencies (non cloud based).
no dependency on commercially licensed software
use a relevant mainstream programming languages (Java, C/C++, Python, R)
the program can be pointed at a file(s) or at a director (and will process every file in that directory), output is written to wherever the program is run from or configurable directory.

# How it works

sample image

![image](https://github.com/user-attachments/assets/57263fef-7af0-49e0-b579-cb37aaef81b9)
![image](https://github.com/user-attachments/assets/64e47b6d-3d04-49fd-9bf8-bd208637432d)
![image](https://github.com/user-attachments/assets/6bfd797d-a919-4ee9-a36f-fcf1d7c86fe2)
![image](https://github.com/user-attachments/assets/25bb2404-5544-45a1-9f0c-a12f20d20dc6)
![image](https://github.com/user-attachments/assets/a9055770-393d-44da-b465-090e22330c13)
![image](https://github.com/user-attachments/assets/ddb8720f-24b2-49ab-8d7a-6b1614329e05)
![image](https://github.com/user-attachments/assets/91e69b50-c7fb-4199-81fc-68b98f036d83)
![image](https://github.com/user-attachments/assets/2cd5c390-4c76-42a7-bd23-d9ec697179ae)
![image](https://github.com/user-attachments/assets/49d3aca5-a486-4420-b6b4-479b8cf70f39)
![image](https://github.com/user-attachments/assets/0c1dc74e-511c-49a5-a8e4-643279885f8a)
![image](https://github.com/user-attachments/assets/b14bda1f-7b68-4c91-b825-b1f2b4bfaa63)
![image](https://github.com/user-attachments/assets/cdc53e12-213d-4806-aff5-9527ce5cc3ce)
![image](https://github.com/user-attachments/assets/4af94e07-d0e7-44f8-a47d-d2b73c0795cd)








