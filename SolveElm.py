# -*- coding: utf-8 -*-
"""
Created on Sat May 28 08:59:37 2022

@author: Administrator
"""

import math
import py2exe
#from ast import If
import numpy as np


def trainElm(num, path):
    d1d2uv = np.loadtxt(path+'/inputnormalized.txt')   #将文件中数据加载到data数组里
    lines = np.loadtxt(path+'/outputnormalized.txt')
    
    if (d1d2uv.shape == 0):
        return 0

    rd = np.random.RandomState(9999)
    
    Iw = rd.random((num,d1d2uv.shape[1]))
    
    B = rd.random((num,1))
    
    Iw = 2*Iw-1
    
    B_repeat = np.tile(B,(1,d1d2uv.shape[0]))
    
    tempH = Iw.dot(d1d2uv.T)+B_repeat
    
    for i in range(tempH.shape[0]):
        for j in range(tempH.shape[1]):
            tempH[i][j] = 1/(math.exp(-1*tempH[i][j])+1)
    
    H_pinv = np.linalg.pinv(tempH)
    
    Lw = H_pinv.T.dot(lines)
    
    file_Iw = open(path + '/ModelParameter/IW.txt', 'w',encoding='UTF-8')
    for i in range (Iw.shape[0]):
        for j in range (Iw.shape[1]):
            file_Iw.write(str(Iw[i][j])+' ')
        file_Iw.write('\n')
    file_Iw.close()
    
    file_B = open(path + '/ModelParameter/B.txt', 'w',encoding='UTF-8')
    for i in range (B.shape[0]):
        for j in range (B.shape[1]):
            file_B.write(str(B[i][j])+' ')
        file_B.write('\n')
    file_B.close()
    
    file_Lw = open(path + '/ModelParameter/LW.txt', 'w',encoding='UTF-8')
    for i in range (Lw.shape[0]):
        for j in range (Lw.shape[1]):
            file_Lw.write(str(Lw[i][j])+' ')
        file_Lw.write('\n')
    file_Lw.close()
    return 1


if __name__ == "__main__":  
    trainElm(3000,'./data/')