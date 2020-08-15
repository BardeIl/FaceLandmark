#coding:utf-8 
import sys
import numpy as np
import cv2
import os
import numpy.random as npr
import cPickle as pickle#python2.7

'''
function prepare
'''
def IoU(box, boxes):
    """Compute IoU between detect box and gt boxes

    Parameters:
    ----------
    box: numpy array , shape (5, ): x1, y1, x2, y2, score
        input box
    boxes: numpy array, shape (n, 4): x1, y1, x2, y2
        input ground truth boxes

    Returns:
    -------
    ovr: numpy.array, shape (n, )
        IoU
    """
    box_area = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    area = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])

    # compute the width and height of the bounding box
    w = np.maximum(0, xx2 - xx1 + 1)
    h = np.maximum(0, yy2 - yy1 + 1)

    inter = w * h
    ovr = inter / (box_area + area - inter)
    return ovr


def convert_to_square(bbox):
    """Convert bbox to square

    Parameters:
    ----------
    bbox: numpy array , shape n x 5
        input bbox

    Returns:
    -------
    square bbox
    """
    square_bbox = bbox.copy()

    h = bbox[:, 3] - bbox[:, 1] + 1
    w = bbox[:, 2] - bbox[:, 0] + 1
    max_side = np.maximum(h,w)
    square_bbox[:, 0] = bbox[:, 0] + w*0.5 - max_side*0.5
    square_bbox[:, 1] = bbox[:, 1] + h*0.5 - max_side*0.5
    square_bbox[:, 2] = square_bbox[:, 0] + max_side - 1
    square_bbox[:, 3] = square_bbox[:, 1] + max_side - 1
    return square_bbox

def ensure_directory_exists(path):
    if not os.path.exists(path):
        os.mkdir(path)

anno_file = "./label.txt"
im_dir = "/home/passwd123/work/widerface/WIDER_train/images/"
img_save_dir = "./64/trainimg/"
label_save_dir = "./64/trainlabel/"
save_dir = "./64/"

ensure_directory_exists(save_dir)
ensure_directory_exists(img_save_dir)

f1 = open(os.path.join(save_dir, 'trainlist.txt'), 'w')
with open(anno_file, 'r') as f:
    annotations = f.readlines()
num = len(annotations)
num = num - 12880
#print ("%d pics in total" % (num))
img_idx = 0 # positive
idx = 0
box_idx = 0
cls_list = []
test_list = []
'''
img1 = cv2.imread("E:/zhonghangyou_dataset/retinaface/wider_face/WIDER_train/images/0--Parade/0_Parade_marchingband_1_849.jpg")
cv2.imshow('t',img1)
cv2.waitKey(0)
'''
for  annotation in annotations:
    annotation = annotation.strip()#移除字符串头尾指定的字符（默认为空格或换行符）.split('/n')
    #print(annotation)
    if(annotation[0]=='#'):
        #print(annotation)
        #print('1:',annotation[1])
        im_path = annotation.strip('# ')
        idx = idx+1
        print(idx)
        #f1.write(im_path+'\n')
        #print(im_path)
        img = cv2.imread(os.path.join(im_dir, im_path))
        imn=0
        #cv2.imshow(im_path,img)
        #cv2.waitKey(0)
    else:
        #print('0')
        im_prepath = im_path.strip('.jpg')
        #print(im_prepath)
        words = annotation.split()
        #print(len(words))
        #print(words[0])
        thres1 = 0
        x1 = int(words[0])
        y1 = int(words[1])
        w = int(words[2])
        h = int(words[3])
        l1 = max(w,h)
        #print(l1)
        

        ptsx = [float(words[4]),float(words[7]),float(words[10]),float(words[13]), float(words[16])]
        ptsy = [float(words[5]),float(words[8]),float(words[11]),float(words[14]), float(words[17])]
        for i in range(0,5):
            #print(ptsx[i],ptsy[i])
            thres1 = thres1 + ptsx[i] + ptsy[i]

        #pts00 = float(words[4])
        #pts01 = float(words[5])
        #pts10 = float(words[7])
        #pts11 = float(words[8])
        #pts20 = float(words[10])
        #pts21 = float(words[11])
        #pts30 = float(words[13])
        #pts31 = float(words[14])
        #pts40 = float(words[16])
        #pts41 = float(words[17])
        #print(x1,y1,w,h,pts00,pts01,pts10,pts11,pts20,pts21,pts30,pts31,pts40,pts41)

        if(l1<50) or (thres1 == -10.0):
            #print('skipe')
            continue
        #print(l1)
        #print('thres1',thres1)
        #cv2.rectangle(img, (x1,y1), (x1+l1,y1+l1), (0,255,0), 4)
        #for i in range(0,5):
            #cv2.circle(img, (int(ptsx[i]),int(ptsy[i])), 1, (0, 0, 255), 4)
        #cv2.imshow(im_path,img)
        cropped_im = img[y1 : (y1+l1), x1 : (x1+l1), :]
        #cv2.imshow('crop',cropped_im)
        resized_im = cv2.resize(cropped_im, (64, 64), interpolation=cv2.INTER_LINEAR)
        imgname = img_save_dir+ im_prepath + '_' + str(imn) + '.jpg'
        print(imgname)
        f1.write(imgname+'\n')
        cv2.imwrite(imgname, resized_im)
        imn = imn+1
        #cv2.imshow('crop&resize',resized_im)
        #cv2.waitKey(0)
        im = np.swapaxes(resized_im, 0, 2)#BGR2RGB
        im = (im - 127.5)/127.5
        for i in range(0,5):
            ptsx[i] = (ptsx[i]-x1)/l1
            ptsy[i] = (ptsy[i]-y1)/l1
        for i in range(0,5):
            print(ptsx[i],ptsy[i])
        pts    = [ptsx[0],ptsy[0],ptsx[1],ptsy[1],ptsx[2],ptsy[2],ptsx[3],ptsy[3],ptsx[4],ptsy[4]]
        cls_list.append([im,pts])
        test_list.append([im])
        #print(pts)
        break

#fid = open("./64/trainnormal.imdb",'wb')
#print(cls_list)
#pickle.dump(cls_list, fid)
#fid.close()
print('done')

#######code test#####
#fid = open("./64/test.imdb",'rb+')
#t = pickle.load(fid)
#print(t[0][1])
#pickle.dump(test_list, fid)
#fid.close()

