# 1. Introduction
Use Python to design an analysis tool with interface scene adaptation capabilities, identify interface operation processes and record operation information.
# 2. Question Requirements
## 1. Basic Requirements
(1) Identification: Use intelligent algorithms to correctly identify the process information of the mouse clicking the interface controls in the given video. The interface controls include buttons, checkboxes, radio buttons, and tool strips;
(2) Recording: Record the data of the mouse clicking operation in the video. The recorded data mainly includes time (required to be accurate to seconds and the time data delay is <1 second) and the name of the clicked control, such as: Button control record---April 16, 2023 08:34:12.000 Click the button "Open Serial Port"; Checkbox control record---April 16, 2023 08:34:12.000 Check the checkbox "Hexadecimal Display"; Radio button control record---April 16, 2023 08:34:12.000 Select the radio button "Working Status"; Toolbar Control Record---April 16, 2023 08:34:12.000 Select the toolbar "Window";
(3) Display: Real-time display of click operation data in the video;
(4) Real-time: The recognition operation rate is not less than 10 frames (10 pictures) per second.
## 2. Performance
(1) Comparison: Compare the recorded data (only for the control name) with the pre-provided record file.txt one by one. If the data matches, the graphical user interface operation is effective. If it does not match, the operation is not effective;
(2) Accuracy: The control name accuracy (compared with the pre-provided record file.txt) is >90%;
(3) Real-time: The recognition rate is not less than 20 frames (20 pictures) per second.
## 3. Report
A report must be submitted. The main content of the report can be found in the scoring criteria
