# GCIS-Calibration
The open source code (in C++), as well as an executable file, 
are provided in this dir. 

**************************
The  "Realease.7z" includes the executable file, necessary dlls and calibration data.

The image points and spatial points of markers are decoded, the results are stored in "./Release/data/ImgInfo.xml" and "./Release/data/PtsInfo.xml" specificly.

You can just open the "./Release/GCIS_calibration_opensource.exe" and input the data path "./data" to test the calibration code.

**************************
The  "GCIS_calibration_opensource.7z" includes the source code of GCIS calibration.

It needs some necessary libs to build the source code, including:
ceres
eigen
opencv
python (with numpy)
