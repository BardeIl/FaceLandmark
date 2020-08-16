#coding:utf-8 
import sys
import numpy as np
import cv2
import os
import numpy.random as npr
import cPickle as pickle#python2.7

fid = open("64/test.imdb",'rb+')
#fid = open("64/train.imdb",'rb+')
t = pickle.load(fid)
#pickle.dump(cls_list, fid)
#print(t[0][0])
#print(t[0][1])
fid.close()


img = cv2.imread('./64/trainimg/0--Parade/0_Parade_marchingband_1_849_0.jpg')
cv2.imshow('test',img)
gt1 = [17.140832214765105, 18.745986577181196,39.9845369127517, 19.948241610738258,28.362308724832193, 35.5779865771812,15.537825503355695,
 41.188939597315425,38.38153020134227, 43.59344966442952]
lk1 = [18.214218 ,23.485262, 34.58618 , 22.582582, 27.202028, 35.2417,   20.74202,
  46.12909,  33.757153 ,45.366776]
lk2 = [15.932087, 18.698034, 40.262863, 20.215227, 28.030174, 33.3858 ,  16.014633,
  42.51534 , 36.54935,  43.728786]
for i in range(0,5):
    cv2.circle(img, (int(gt1[i*2]),int(gt1[i*2+1])), 1, (0, 0, 255), 1)
for i in range(0,5):
    cv2.circle(img, (int(lk1[i*2]),int(lk1[i*2+1])), 1, (0, 255, 0), 1)
for i in range(0,5):
    cv2.circle(img, (int(lk2[i*2]),int(lk2[i*2+1])), 1, (255, 0, 0), 1)
img = cv2.resize( img, (128, 128), interpolation=cv2.INTER_LINEAR)
cv2.imshow('test',img)
cv2.waitKey(0)