from keras.layers import Input, Dense, Conv2D,BatchNormalization,Lambda
from keras.models import Model, load_model
import numpy as np
import tensorflow as tf
import cv2
import time

##input###

'''place input here (16,16,1)'''

start_time = time.time()

x_test1 = cv2.imread('t256.png')
imgpatch = np.zeros((125*125 ,16,16,1))

k = 0
for i in range(125):
  for j in range(125):

    img_patch0= x_test1[i*4:i*4+16,j*4:j*4+16,0]
    ###h, w = img_patch0.shape
    ###print(h, " ", w)
    imgpatch[k,:,:,0] = img_patch0
    k=k+1

print("test set is created")

###Autoenc Arch###
input_img = Input(shape=(16, 16,1))

autoencoder = load_model('trained_dcae.h5', custom_objects={'tf': tf})

imgpatch = imgpatch.astype('float32') /255.
yy= autoencoder.predict(imgpatch)
yy=yy*255.

print("end of prediction")

print("--- %s seconds ---" % (time.time() - start_time))

x_test1=x_test1.astype('float32')
bn = np.zeros((512,512,1))
kn = np.zeros((512,512,1))

k = 0

for i in range(125):
  for j in range(125):

    i0= x_test1[i*4:i*4+16,j*4:j*4+16,0]
    i1= yy[k,:,:,0]

    bn[i*4:i*4+16,j*4:j*4+16,0]=bn[i*4:i*4+16,j*4:j*4+16,0]+i1
    kn[i*4:i*4+16,j*4:j*4+16,0]=kn[i*4:i*4+16,j*4:j*4+16,0]+1

    k=k+1

bn=bn/kn

bn=bn.astype(np.uint8)

cv2.imshow('output',bn)
cv2.imwrite('output256.png',bn)
cv2.waitKey()
