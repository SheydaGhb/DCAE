from keras.layers import Input, Dense, Conv2D,BatchNormalization,Lambda,MaxPooling2D,UpSampling2D
from keras.models import Model
from keras import backend as K
import numpy as np
import tensorflow as tf
import scipy.io
from keras.callbacks import LearningRateScheduler, TensorBoard,ModelCheckpoint
from keras import regularizers
import cv2
from keras.models import load_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import time





####imread train 1million input

print('start')
mat1 = scipy.io.loadmat('A1.mat')
mat2 = scipy.io.loadmat('A2.mat')
mat3 = scipy.io.loadmat('A3.mat')


A1=mat1.get('A1')
A2=mat2.get('A2')
A3=mat3.get('A3')


x_train=np.concatenate((A1,A2,A3), axis =1) ## concatinate all image patches of size 16x16



img_patch = np.zeros(((A1[0,0].shape[2]*A1.shape[1])+((A2[0,0].shape[2])*A2.shape[1])+(A3[0,0].shape[2]*A3.shape[1]),16,16))

k=0
for i in range (0,x_train.shape[1]):
    for j in range(0,x_train[0,i].shape[2]):
        img_patch[k,:,:]=x_train[0,i][:,:,j]

        k=k+1


img_patch =  np.expand_dims(img_patch, axis=3)
#print(img_patch.shape)



### autoencoder structure###
start_time = time.time()
input_img = Input(shape=(16, 16,1))

paddings = tf.constant([ [0,0],[1, 1], [1, 1], [0,0]])

x = Lambda(lambda y: tf.pad(y , paddings,  "SYMMETRIC"))(input_img)
x = Conv2D(256, (3, 3), activation='relu' , padding = 'valid')(x)
x = Lambda(lambda y: tf.pad(y , paddings,  "SYMMETRIC"))(x)
x = Conv2D(128, (3, 3), activation='relu', padding='valid')(x)
x= BatchNormalization()(x)
x = Lambda(lambda y: tf.pad(y , paddings,  "SYMMETRIC"))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='valid')(x)
x = Lambda(lambda y: tf.space_to_depth(y , 2))(x)
x = Lambda(lambda y: tf.pad(y , paddings,  "SYMMETRIC"))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='valid')(x)
x = Lambda(lambda y: tf.pad(y , paddings,  "SYMMETRIC"))(x)
x = Conv2D(16, (3, 3), activation='relu', padding='valid')(x)
x= BatchNormalization()(x)
x = Lambda(lambda y: tf.pad(y , paddings,  "SYMMETRIC"))(x)
x = Conv2D(4, (3, 3), activation='relu', padding='valid')(x)
x = Lambda(lambda y: tf.pad(y , paddings,  "SYMMETRIC"))(x)
encoded = Conv2D(1, (3, 3), activation='sigmoid', padding='valid')(x)


x = Lambda(lambda y: tf.pad(y , paddings,  "SYMMETRIC"))(encoded)
x = Conv2D(64, (3, 3), activation='relu', padding='valid')(x)
x = Lambda(lambda y: tf.pad(y , paddings,  "SYMMETRIC"))(x)
x = Conv2D(256, (3, 3), activation='relu', padding='valid')(x)
x= BatchNormalization()(x)
x = Lambda(lambda y: tf.depth_to_space(y , 2))(x)
x = Lambda(lambda y: tf.pad(y , paddings,  "SYMMETRIC"))(x)
x = Conv2D(256, (3, 3), activation='relu' ,padding = 'valid' )(x)
x = Lambda(lambda y: tf.pad(y , paddings,  "SYMMETRIC"))(x)
x = Conv2D(16, (3, 3), activation='relu' ,padding = 'valid' )(x)
x= BatchNormalization()(x)
x = Lambda(lambda y: tf.pad(y , paddings,  "SYMMETRIC"))(x)
x = Conv2D(4, (3, 3), activation='relu' ,padding = 'valid' )(x)
x = Lambda(lambda y: tf.pad(y, paddings,  "SYMMETRIC"))(x)
decoded = Conv2D(1, (3, 3), activation='sigmoid' ,padding = 'valid' )(x)


autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer= 'adam', loss='mse')
autoencoder.summary()

##uncomment if it is contiutaion of train from a checkpoint model

'''filepath="ckpnt_model.h5"
checkpoint= ModelCheckpoint(filepath, monitor='loss' , verbose=1 , mode='min')
callback_list = [checkpoint]'''



img_patch = img_patch.astype('float32') /255.
model_history = autoencoder.fit(img_patch, img_patch,
                epochs=500,
                batch_size=2048,
				#callbacks =callback_list,  ##uncomment if it is contiutaion of train from a checkpoint model
                 )

autoencoder.save('DCAE.h5')
autoencoder.save_weights('DCAE_weights.h5')


print("--- %s minutes ---" % ((time.time() - start_time)/60))

hist= model_history.history['loss']
np.save('DCAE_history.npy', hist)


