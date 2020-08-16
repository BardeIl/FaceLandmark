#coding:utf-8 
import os
import sys
caffe_root = '/home/passwd123/work/caffe/'
sys.path.append(caffe_root+'python')#
import cv2
import caffe
import numpy as np
import matplotlib.pyplot as plt

# set display defaults
plt.rcParams['figure.figsize'] = (64, 64)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'color'  # use grayscale output rather than a (potentially misleading) color heatmap

#loading caffe model
model_path = '/home/passwd123/work/mtcnn-caffe-master/64net/'
#prototxt_path = model_path + '64net.prototxt'
#caffemodel_path = model_path + 'models/solver_iter_600000.caffemodel'#'weights/solver_iter_120000.caffemodel'
prototxt_path = model_path + '64origin.prototxt'
caffemodel_path = model_path + 'modelsorigin/solver_iter_600000.caffemodel'#'weights/solver_iter_120000.caffemodel'
if os.path.isfile(prototxt_path):
    print 'Caffe model found.'
else:
    print 'Caffe model not found, please check....'
if os.path.isfile(caffemodel_path):
    print 'model weights found.'
else:
    print 'Caffe model weights not found, please check....'



#caffe.set_mode_cpu()
caffe.set_device(0)  # if we have multiple GPUs, pick the first one
caffe.set_mode_gpu()

model_def = prototxt_path
model_weights = caffemodel_path

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# create transformer for the input called 'data'
#transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

#mu = [0.5,0.5,0.5]

#transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
#transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
#transformer.set_raw_scale('data', 1)      # rescale from [0, 1] to [0, 255]
#transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

in_size = 64
net.blobs['data'].reshape(1,        # batch size
                          3,         # 3-channel (BGR) images
                          in_size, in_size)  # image size is 64x64

img_path = "/home/passwd123/work/mtcnn-caffe-master/64net/64/trainimg/0--Parade/0_Parade_marchingband_1_849_0.jpg"
im = cv2.imread(img_path)
im = np.swapaxes(im, 0, 2)
im = (im - 127.5)/127.5
#image = caffe.io.load_image(img_path)
#transformed_image = transformer.preprocess('data', image)
#plt.imshow(image)

net.blobs['data'].data[...] = im

### perform classification
output = net.forward()

output_prob = output['out']#[1]  # the output probability vector for the first image in the batch

gt1 = [17.140832214765105, 18.745986577181196,39.9845369127517, 19.948241610738258,28.362308724832193, 35.5779865771812,15.537825503355695, 41.188939597315425,38.38153020134227, 43.59344966442952]
gt2 = [0.26782550335570476, 0.2929060402684562,0.6247583892617453, 0.3116912751677853,0.443161073825503, 0.5559060402684562,0.24277852348993273, 0.6435771812080535,0.599711409395973, 0.6811476510067113]

print 'gt1:',gt1
print 'gt2:',gt2
print 'predicted pts is:', output_prob
outb = output_prob[0]
img = cv2.imread('./64/trainimg/0--Parade/0_Parade_marchingband_1_849_0.jpg')
cv2.imshow('test',img)
for i in range(0,5):
    cv2.circle(img, (int(gt1[i*2]),int(gt1[i*2+1])), 1, (0, 0, 255), 1)
for i in range(0,5):
    cv2.circle(img, (int(outb[i*2]),int(outb[i*2+1])), 1, (255, 0, 0), 1)
img = cv2.resize( img, (128, 128), interpolation=cv2.INTER_LINEAR)
cv2.imshow('test',img)
cv2.waitKey(0)

if __name__=="__main__": 
    print "testing trained caffe model"
