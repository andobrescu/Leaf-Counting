from __future__ import division, print_function, absolute_import

import os
import traceback
import keras
import keras.backend as K
from keras.activations import sigmoid
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Dropout, Flatten, merge, ConvLSTM2D, Reshape
from keras.layers import Input, Convolution2D, MaxPooling2D, LeakyReLU, LSTM, TimeDistributed, Conv2D
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras import metrics
from keras import backend as K
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.applications.resnet50 import ResNet50
# from Resnet50_keras import ResNet50
from sklearn.metrics import r2_score, mean_squared_error
from datetime import datetime
from collections import OrderedDict
import scipy.misc as misc
import matplotlib.pyplot as plt
import numpy as np
import h5py
import glob
import pandas as pd
import random
from PIL import Image, ImageOps


# Splits for train, test, validation
split_1 = [[1,2], 3, 4]
split_2 = [[2,3], 4, 1]
split_3 = [[3,4], 1, 2]
split_4 = [[4,1], 2, 3]

splits = {1 : split_1,
		  2 : split_2,
		  3 : split_3,
		  4 : split_4,
		 }
split_idx = list(range(1,4))
random_split = random.sample(split_idx, 1)

# Choosing which split to do or all
while True:
	while True:
		split_decision = input('Type to choose a split or run all splits. (1/2/3/4/all):   ')
		if split_decision in ('1','2','3','4','all'):
			break
		print('Invalid input. Choose split 1,2,3,4 or all')

	if split_decision == '1':
		split_load = splits[1]
	elif split_decision == '2':
		split_load = splits[2]
	elif split_decision == '3':
		split_load = splits[3]
	elif split_decision == '4':
		split_load = splits[4]
	elif split_decision == 'all':
		split_all = [3,4]
	break


# Loading CVPPP 2017 data from files
def get_data(split_load):
	####################################
	# Getting images (x data)	  	   
	# *_img = image name
	# *_set = which set it comes from
	####################################
	imgname_train_A1 = np.array([glob.glob('CVPPP2017_LCC_training/TrainingSplits/A1'+str(h)+'/*rgb.png') for h in split_load[0]])
	imgname_train_A2 = np.array([glob.glob('CVPPP2017_LCC_training/TrainingSplits/A2'+str(h)+'/*rgb.png') for h in split_load[0]])
	imgname_train_A3 = np.array([glob.glob('CVPPP2017_LCC_training/TrainingSplits/A3'+str(h)+'/*rgb.png') for h in split_load[0]])
	imgname_train_A4 = np.array([glob.glob('CVPPP2017_LCC_training/TrainingSplits/A4'+str(h)+'/*rgb.png') for h in split_load[0]])

	imgname_val_A1 = np.array([glob.glob('CVPPP2017_LCC_training/TrainingSplits/A1'+str(split_load[1])+'/*rgb.png')])
	imgname_val_A2 = np.array([glob.glob('CVPPP2017_LCC_training/TrainingSplits/A2'+str(split_load[1])+'/*rgb.png')])
	imgname_val_A3 = np.array([glob.glob('CVPPP2017_LCC_training/TrainingSplits/A3'+str(split_load[1])+'/*rgb.png')])
	imgname_val_A4 = np.array([glob.glob('CVPPP2017_LCC_training/TrainingSplits/A4'+str(split_load[1])+'/*rgb.png')])

	imgname_test_A1 = np.array([glob.glob('CVPPP2017_LCC_training/TrainingSplits/A1'+str(split_load[2])+'/*rgb.png')])
	imgname_test_A2 = np.array([glob.glob('CVPPP2017_LCC_training/TrainingSplits/A2'+str(split_load[2])+'/*rgb.png')])
	imgname_test_A3 = np.array([glob.glob('CVPPP2017_LCC_training/TrainingSplits/A3'+str(split_load[2])+'/*rgb.png')])
	imgname_test_A4 = np.array([glob.glob('CVPPP2017_LCC_training/TrainingSplits/A4'+str(split_load[2])+'/*rgb.png')])

	# List with all the images of training images for the sets
	filelist_train_A1 = list(np.sort(imgname_train_A1.flat))
	filelist_train_A2 = list(np.sort(imgname_train_A2.flat))
	filelist_train_A3 = list(np.sort(imgname_train_A3.flat))
	filelist_train_A4 = list(np.sort(imgname_train_A4.flat))

	# Get Image name for training all the sets
	filelist_train_A1_img = np.array([np.array(filelist_train_A1[h][42:58]) for h in range(0,len(filelist_train_A1))])
	filelist_train_A2_img = np.array([np.array(filelist_train_A2[h][42:58]) for h in range(0,len(filelist_train_A2))])
	filelist_train_A3_img = np.array([np.array(filelist_train_A3[h][42:58]) for h in range(0,len(filelist_train_A3))])
	filelist_train_A4_img = np.array([np.array(filelist_train_A4[h][42:59]) for h in range(0,len(filelist_train_A4))])

	# Get Set name for training all the sets
	filelist_train_A1_set = np.array([np.array(filelist_train_A1[h][38:40]) for h in range(0,len(filelist_train_A1))])
	filelist_train_A2_set = np.array([np.array(filelist_train_A2[h][38:40]) for h in range(0,len(filelist_train_A2))])
	filelist_train_A3_set = np.array([np.array(filelist_train_A3[h][38:40]) for h in range(0,len(filelist_train_A3))])
	filelist_train_A4_set = np.array([np.array(filelist_train_A4[h][38:40]) for h in range(0,len(filelist_train_A4))])

	# List with all the images of validation images for the sets
	filelist_val_A1 = list(np.sort(imgname_val_A1.flat))
	filelist_val_A2 = list(np.sort(imgname_val_A2.flat))
	filelist_val_A3 = list(np.sort(imgname_val_A3.flat))
	filelist_val_A4 = list(np.sort(imgname_val_A4.flat))

	# Get Image name for validation all the sets	
	filelist_val_A1_img = np.array([np.array(filelist_val_A1[h][42:58]) for h in range(0,len(filelist_val_A1))])
	filelist_val_A2_img = np.array([np.array(filelist_val_A2[h][42:58]) for h in range(0,len(filelist_val_A2))])
	filelist_val_A3_img = np.array([np.array(filelist_val_A3[h][42:58]) for h in range(0,len(filelist_val_A3))])
	filelist_val_A4_img = np.array([np.array(filelist_val_A4[h][42:59]) for h in range(0,len(filelist_val_A4))])

	# Get Set name for validation all the sets
	filelist_val_A1_set = np.array([np.array(filelist_val_A1[h][38:40]) for h in range(0,len(filelist_val_A1))])
	filelist_val_A2_set = np.array([np.array(filelist_val_A2[h][38:40]) for h in range(0,len(filelist_val_A2))])
	filelist_val_A3_set = np.array([np.array(filelist_val_A3[h][38:40]) for h in range(0,len(filelist_val_A3))])
	filelist_val_A4_set = np.array([np.array(filelist_val_A4[h][38:40]) for h in range(0,len(filelist_val_A4))])

	# List with all the images of test images for the sets
	filelist_test_A1 = list(np.sort(imgname_test_A1.flat))
	filelist_test_A2 = list(np.sort(imgname_test_A2.flat))
	filelist_test_A3 = list(np.sort(imgname_test_A3.flat))
	filelist_test_A4 = list(np.sort(imgname_test_A4.flat))

	# Get Image name for test all the sets
	filelist_test_A1_img = np.array([np.array(filelist_test_A1[h][42:58]) for h in range(0,len(filelist_test_A1))])
	filelist_test_A2_img = np.array([np.array(filelist_test_A2[h][42:58]) for h in range(0,len(filelist_test_A2))])
	filelist_test_A3_img = np.array([np.array(filelist_test_A3[h][42:58]) for h in range(0,len(filelist_test_A3))])
	filelist_test_A4_img = np.array([np.array(filelist_test_A4[h][42:59]) for h in range(0,len(filelist_test_A4))])
	
	# Get Set name for test all the sets
	filelist_test_A1_set = np.array([np.array(filelist_test_A1[h][38:40]) for h in range(0,len(filelist_test_A1))])
	filelist_test_A2_set = np.array([np.array(filelist_test_A2[h][38:40]) for h in range(0,len(filelist_test_A2))])
	filelist_test_A3_set = np.array([np.array(filelist_test_A3[h][38:40]) for h in range(0,len(filelist_test_A3))])
	filelist_test_A4_set = np.array([np.array(filelist_test_A4[h][38:40]) for h in range(0,len(filelist_test_A4))])

	#Read image names into np array train
	x_train_A1 = np.array([np.array(misc.imread(fname)) for fname in filelist_train_A1])
	x_train_A1 = np.delete(x_train_A1,3,3)
	x_train_A2 = np.array([np.array(misc.imread(fname)) for fname in filelist_train_A2])
	x_train_A2 = np.delete(x_train_A2,3,3)
	x_train_A3 = np.array([np.array(Image.open(fname)) for fname in filelist_train_A3])
	x_train_A4 = np.array([np.array(Image.open(fname)) for fname in filelist_train_A4])

	#Read image names into np array validation
	x_val_A1 = np.array([np.array(misc.imread(fname)) for fname in filelist_val_A1])
	x_val_A1 = np.delete(x_val_A1,3,3)
	x_val_A2 = np.array([np.array(misc.imread(fname)) for fname in filelist_val_A2])
	x_val_A2 = np.delete(x_val_A2,3,3)
	x_val_A3 = np.array([np.array(Image.open(fname)) for fname in filelist_val_A3])
	x_val_A4 = np.array([np.array(Image.open(fname)) for fname in filelist_val_A4])

	#Read image names into np array test
	x_test_A1 = np.array([np.array(misc.imread(fname)) for fname in filelist_test_A1])
	x_test_A1 = np.delete(x_test_A1,3,3)
	x_test_A2 = np.array([np.array(Image.open(fname)) for fname in filelist_test_A2])
	x_test_A2 = np.delete(x_test_A2,3,3)
	x_test_A3 = np.array([np.array(Image.open(fname)) for fname in filelist_test_A3])
	x_test_A4 = np.array([np.array(Image.open(fname)) for fname in filelist_test_A4])

	#Resize images to make them all the same size
	x_train_res_A1 = np.array([misc.imresize(x_train_A1[i],[320,320,3]) for i in range(0,len(x_train_A1))])
	x_train_res_A2 = np.array([misc.imresize(x_train_A2[i],[320,320,3]) for i in range(0,len(x_train_A2))])
	x_train_res_A3 = np.array([misc.imresize(x_train_A3[i],[320,320,3]) for i in range(0,len(x_train_A3))])
	x_train_res_A4 = np.array([misc.imresize(x_train_A4[i],[320,320,3]) for i in range(0,len(x_train_A4))])

	x_val_res_A1 = np.array([misc.imresize(x_val_A1[i],[320,320,3]) for i in range(0,len(x_val_A1))])
	x_val_res_A2 = np.array([misc.imresize(x_val_A2[i],[320,320,3]) for i in range(0,len(x_val_A2))])
	x_val_res_A3 = np.array([misc.imresize(x_val_A3[i],[320,320,3]) for i in range(0,len(x_val_A3))])
	x_val_res_A4 = np.array([misc.imresize(x_val_A4[i],[320,320,3]) for i in range(0,len(x_val_A4))])

	x_test_res_A1 = np.array([misc.imresize(x_test_A1[i],[320,320,3]) for i in range(0,len(x_test_A1))])
	x_test_res_A2 = np.array([misc.imresize(x_test_A2[i],[320,320,3]) for i in range(0,len(x_test_A2))])
	x_test_res_A3 = np.array([misc.imresize(x_test_A3[i],[320,320,3]) for i in range(0,len(x_test_A3))])
	x_test_res_A4 = np.array([misc.imresize(x_test_A4[i],[320,320,3]) for i in range(0,len(x_test_A4))])

	#Concatenate the sets into one array
	x_train_all = np.concatenate((x_train_res_A1, x_train_res_A2, x_train_res_A3, x_train_res_A4), axis=0)
	x_val_all = np.concatenate((x_val_res_A1, x_val_res_A2, x_val_res_A3, x_val_res_A4), axis=0)
	x_test_all = np.concatenate((x_test_res_A1, x_test_res_A2, x_test_res_A3, x_test_res_A4), axis=0)

	#Histogram stretching
	for h in range(0,len(x_train_all)):
		x_img = x_train_all[h]
		x_img_pil = Image.fromarray(x_img)
		x_img_pil = ImageOps.autocontrast(x_img_pil)
		x_img_ar = np.array(x_img_pil)
		x_train_all[h] = x_img_ar

	for h in range(0,len(x_val_all)):
		x_img = x_val_all[h]
		x_img_pil = Image.fromarray(x_img)
		x_img_pil = ImageOps.autocontrast(x_img_pil)
		x_img_ar = np.array(x_img_pil)
		x_val_all[h] = x_img_ar

	for h in range(0,len(x_test_all)):
		x_img = x_test_all[h]
		x_img_pil = Image.fromarray(x_img)
		x_img_pil = ImageOps.autocontrast(x_img_pil)
		x_img_ar = np.array(x_img_pil)
		x_test_all[h] = x_img_ar

	# Concatenate the image names
	x_train_img = np.concatenate((filelist_train_A1_img, filelist_train_A2_img, filelist_train_A3_img, filelist_train_A4_img), axis=0)
	x_val_img = np.concatenate((filelist_val_A1_img, filelist_val_A2_img, filelist_val_A3_img, filelist_val_A4_img), axis=0)
	x_test_img = np.concatenate((filelist_test_A1_img, filelist_test_A2_img, filelist_test_A3_img, filelist_test_A4_img), axis=0)

	#Concatenate the set names
	x_train_set = np.concatenate((filelist_train_A1_set, filelist_train_A2_set, filelist_train_A3_set, filelist_train_A4_set), axis=0)
	x_val_set = np.concatenate((filelist_val_A1_set, filelist_val_A2_set, filelist_val_A3_set, filelist_val_A4_set), axis=0)
	x_test_set = np.concatenate((filelist_test_A1_set, filelist_test_A2_set, filelist_test_A3_set, filelist_test_A4_set), axis=0)
	

	###############################
	# Getting targets (y data)	  #
	###############################
	counts_A1 = np.array([glob.glob('CVPPP2017_LCC_training/TrainingSplits/A1.xlsx')])
	counts_A2 = np.array([glob.glob('CVPPP2017_LCC_training/TrainingSplits/A2.xlsx')])
	counts_A3 = np.array([glob.glob('CVPPP2017_LCC_training/TrainingSplits/A3.xlsx')])
	counts_A4 = np.array([glob.glob('CVPPP2017_LCC_training/TrainingSplits/A4.xlsx')])

	#Get labels for set A1
	counts_train_flat_A1 = list(counts_A1.flat)
	train_labels_A1 = pd.DataFrame()
	y_train_A1_list = []
	y_val_A1_list = []
	y_test_A1_list = []
	for f in counts_train_flat_A1:
		frame = pd.read_excel(f, header=None)
		train_labels_A1 = train_labels_A1.append(frame, ignore_index=False)
	all_labels_A1 = np.array(train_labels_A1)

	for j in filelist_train_A1_img:
		arr_idx = np.where(all_labels_A1 == j)
		y_train_A1_list.append(all_labels_A1[arr_idx[0],:])
	y_train_A1_labels = np.concatenate(y_train_A1_list, axis=0)

	for j in filelist_val_A1_img:
		arr_idx = np.where(all_labels_A1 == j)
		y_val_A1_list.append(all_labels_A1[arr_idx[0],:])
	y_val_A1_labels = np.concatenate(y_val_A1_list, axis=0)

	for j in filelist_test_A1_img:
		arr_idx = np.where(all_labels_A1 == j)
		y_test_A1_list.append(all_labels_A1[arr_idx[0],:])
	y_test_A1_labels = np.concatenate(y_test_A1_list, axis=0)

	#Labels for set A2
	counts_train_flat_A2 = list(counts_A2.flat)
	train_labels_A2 = pd.DataFrame()
	y_train_A2_list = []
	y_val_A2_list = []
	y_test_A2_list = []
	for f in counts_train_flat_A2:
		frame = pd.read_excel(f, header=None)
		train_labels_A2 = train_labels_A2.append(frame, ignore_index=False)
	all_labels_A2 = np.array(train_labels_A2)

	for j in filelist_train_A2_img:
		arr_idx = np.where(all_labels_A2 == j)
		y_train_A2_list.append(all_labels_A2[arr_idx[0],:])
	y_train_A2_labels = np.concatenate(y_train_A2_list, axis=0)

	for j in filelist_val_A2_img:
		arr_idx = np.where(all_labels_A2 == j)
		y_val_A2_list.append(all_labels_A2[arr_idx[0],:])
	y_val_A2_labels = np.concatenate(y_val_A2_list, axis=0)

	for j in filelist_test_A2_img:
		arr_idx = np.where(all_labels_A2 == j)
		y_test_A2_list.append(all_labels_A2[arr_idx[0],:])
	y_test_A2_labels = np.concatenate(y_test_A2_list, axis=0)

	#Labels for set A3
	counts_train_flat_A3 = list(counts_A3.flat)
	train_labels_A3 = pd.DataFrame()
	y_train_A3_list = []
	y_val_A3_list = []
	y_test_A3_list = []
	for f in counts_train_flat_A3:
		frame = pd.read_excel(f, header=None)
		train_labels_A3 = train_labels_A3.append(frame, ignore_index=False)
	all_labels_A3 = np.array(train_labels_A3)

	for j in filelist_train_A3_img:
		arr_idx = np.where(all_labels_A3 == j)
		y_train_A3_list.append(all_labels_A3[arr_idx[0],:])
	y_train_A3_labels = np.concatenate(y_train_A3_list, axis=0)

	for j in filelist_val_A3_img:
		arr_idx = np.where(all_labels_A3 == j)
		y_val_A3_list.append(all_labels_A3[arr_idx[0],:])
	y_val_A3_labels = np.concatenate(y_val_A3_list, axis=0)

	for j in filelist_test_A3_img:
		arr_idx = np.where(all_labels_A3 == j)
		y_test_A3_list.append(all_labels_A3[arr_idx[0],:])
	y_test_A3_labels = np.concatenate(y_test_A3_list, axis=0)

	#labels for set A4
	counts_train_flat_A4 = list(counts_A4.flat)
	train_labels_A4 = pd.DataFrame()
	y_train_A4_list = []
	y_val_A4_list = []
	y_test_A4_list = []
	for f in counts_train_flat_A4:
		frame = pd.read_excel(f, header=None)
		train_labels_A4 = train_labels_A4.append(frame, ignore_index=False)
	all_labels_A4 = np.array(train_labels_A4)

	for j in filelist_train_A4_img:
		arr_idx = np.where(all_labels_A4 == j)
		y_train_A4_list.append(all_labels_A4[arr_idx[0],:])
	y_train_A4_labels = np.concatenate(y_train_A4_list, axis=0)

	for j in filelist_val_A4_img:
		arr_idx = np.where(all_labels_A4 == j)
		y_val_A4_list.append(all_labels_A4[arr_idx[0],:])
	y_val_A4_labels = np.concatenate(y_val_A4_list, axis=0)

	for j in filelist_test_A4_img:
		arr_idx = np.where(all_labels_A4 == j)
		y_test_A4_list.append(all_labels_A4[arr_idx[0],:])
	y_test_A4_labels = np.concatenate(y_test_A4_list, axis=0)

	#concatenate the labels
	y_train_all_labels = np.concatenate((y_train_A1_labels, y_train_A2_labels, y_train_A3_labels, y_train_A4_labels), axis=0)
	y_val_all_labels = np.concatenate((y_val_A1_labels, y_val_A2_labels, y_val_A3_labels, y_val_A4_labels), axis=0)
	y_test_all_labels = np.concatenate((y_test_A1_labels, y_test_A2_labels, y_test_A3_labels, y_test_A4_labels), axis=0)

	#Take just teh label numbers
	y_train_all = y_train_all_labels[:,1]
	y_val_all = y_val_all_labels[:,1]
	y_test_all = y_test_all_labels[:,1]

	return x_train_all, x_val_all, x_test_all, y_train_all, y_val_all, y_test_all, x_train_set, x_val_set, x_test_set, x_train_img, x_val_img, x_test_img




epsilon = 1.0e-9


def poisson_loss_custom(y_true, y_pred):
	return K.abs(K.mean(y_pred - y_true * K.log(y_pred + epsilon), axis=-1))

def binary_accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)

def counter_model(x_train_all, x_val_all, y_train_all, y_val_all):
	x_aug = ImageDataGenerator(
	    rotation_range=180,
	    width_shift_range=0.1,
	    height_shift_range=0.1,
	    horizontal_flip=True,
	    vertical_flip=True,
	    )

	# compute quantities required for featurewise normalization
	# (std, mean, and principal components if ZCA whitening is applied)
	x_aug.fit(x_train_all)


	res_model = ResNet50(weights='imagenet', include_top=False, input_shape=(320,320, 3))
	model = res_model.output
	model = Flatten(name='flatten')(model)
	model = Dense(1024, activation='relu')(model)
	model = Dense(512, activation='relu', activity_regularizer=regularizers.l2(0.02))(model)
	leaf_pred = Dense(1)(model)


	eps = 50
	csv_logger = keras.callbacks.CSVLogger(results_path+'/training.log', separator=',')
	early_stop = EarlyStopping(monitor='val_loss', min_delta=0.03, mode='min', patience=8)


	model = Model(inputs = res_model.input, outputs = leaf_pred)



	model.compile(optimizer=Adam(lr=0.0001), loss= 'mse')
	#fitted_model = model.fit(x_train_all, y_train_all, epochs=eps, batch_size=16, validation_split=0.1, callbacks= [csv_logger])
	fitted_model= model.fit_generator(x_aug.flow(x_train_all, y_train_all, batch_size=6), steps_per_epoch=812,
												 epochs=eps, validation_data=(x_val_all, y_val_all), callbacks= [csv_logger, early_stop])
	#model = load_model(results_path+'/the_model.h5')

	## Saving Model parameters
	# for i, layer in enumerate(res_model.layers):
	# 	print(i, layer.name)
	#model = load_model('./Results/PhenotikiCounter/CVPPP_Chal_split2 2017-06-09 16:45:31/the_model.h5')

	model_json = model.to_json()
	model.save(results_path+'/the_model.h5')


	# #Plotting Loss
	# plt.title('Leaf Counting Loss')
	# plt.plot(range(1,eps+1), fitted_model.history['loss'], label='Training', color='k')
	# plt.plot(range(1,eps+1), fitted_model.history['val_loss'], label='Validation', color='r')
	# plt.xticks(range(1,eps+1))
	# plt.xlabel('Epochs')
	# plt.ylabel('Loss')
	# plt.legend(loc='best')
	# plt.savefig(results_path+'/counter_network_train.png')


	return model

def testing_results(model, x_train_all, x_test_all, y_train_all, y_test_all, x_train_img, x_test_img):
	# Training set results
	predictions_train = model.predict(x_train_all)
	predictions_round_train = np.round(predictions_train)
	y_train_arr = np.reshape(y_train_all, (len(y_train_all),1))
	x_train_img_arr = np.reshape(x_train_img, (len(y_train_all),1))
	result_arr_train = np.concatenate((y_train_arr , predictions_round_train, x_train_img_arr, predictions_train), axis=1)

	difference_arr_train = np.array(np.round([(result_arr_train[h,1]-result_arr_train[h,0]) for h in range(0,len(y_train_arr))]))
	difference_arr_train = difference_arr_train.reshape(difference_arr_train.size, 1)
	difference_std_train = np.std(difference_arr_train)
	average_diff_train = np.average(difference_arr_train)

	difference_arr_abs_train = np.array(np.round([abs(result_arr_train[h,1]-result_arr_train[h,0]) for h in range(0,len(y_train_arr))]))
	difference_arr_abs_train = difference_arr_abs_train.reshape(difference_arr_abs_train.size, 1)
	difference_std_abs_train = np.std(difference_arr_abs_train)
	average_diff_abs_train = np.average(difference_arr_abs_train)

	prediction_equal_train = np.equal(result_arr_train[:,0],result_arr_train[:,1])
	prediction_equal_train = prediction_equal_train.astype(int)
	prediction_equal_train = np.reshape(prediction_equal_train, (len(result_arr_train),1))

	result_arr_train = np.concatenate((result_arr_train,prediction_equal_train), axis=1)

	mean_arr_train = np.mean(result_arr_train[:,1], axis=0)
	r_coeff_train = r2_score(y_train_arr, predictions_round_train)
	MSE_train = mean_squared_error(y_train_arr,predictions_round_train)
	agreement_sum_train = np.sum(prediction_equal_train)
	agreement_train = agreement_sum_train/len(y_train_all)

	# Test set results
	predictions = model.predict(x_test_all)
	predictions_round = np.round(predictions)
	y_test_arr = np.reshape(y_test_all, (len(y_test_all),1))
	x_test_img_arr = np.reshape(x_test_img, (len(y_test_all),1))
	result_arr = np.concatenate((y_test_arr , predictions_round, x_test_img_arr, predictions), axis=1)

	# ##### Tests
	# predictions_masked = model.predict(x_test_masked)
	# predictions_round_masked = np.round(predictions_masked)
	# y_test_arr_masked = np.reshape(y_test, (len(y_test),1))
	# result_arr_masked = np.concatenate((y_test_arr_masked , predictions_round_masked), axis=1)

	# #####

	difference_arr = np.array(np.round([(result_arr[h,1]-result_arr[h,0]) for h in range(0,len(y_test_arr))]))
	difference_arr = difference_arr.reshape(difference_arr.size, 1)
	difference_std = np.std(difference_arr)
	average_diff = np.average(difference_arr)

	difference_arr_abs = np.array(np.round([abs(result_arr[h,1]-result_arr[h,0]) for h in range(0,len(y_test_arr))]))
	difference_arr_abs = difference_arr_abs.reshape(difference_arr_abs.size, 1)
	difference_std_abs = np.std(difference_arr_abs)
	average_diff_abs = np.average(difference_arr_abs)

	prediction_equal = np.equal(result_arr[:,0],result_arr[:,1])
	prediction_equal = prediction_equal.astype(int)
	prediction_equal = np.reshape(prediction_equal, (len(result_arr),1))

	result_arr = np.concatenate((result_arr,prediction_equal), axis=1)


	mean_arr = np.mean(result_arr[:,1], axis=0) #Means
	r_coeff = r2_score(y_test_arr, predictions_round) # R^2 coefficient of test results
	MSE = mean_squared_error(y_test_arr,predictions_round) # MSE of test results
	agreement_sum = np.sum(prediction_equal)
	agreement = agreement_sum/len(y_test_arr) # percent agreement


	# Results dataframe
	results_dict = {'DIC train':[average_diff_train], 'STD DIC train':[difference_std_train], 
					'|DIC| train':[average_diff_abs_train], 'STD |DIC| train':[difference_std_abs_train],
					'MSE train':[MSE_train], 'R^2 train':[r_coeff_train],
					'DIC':[average_diff], 'STD DIC':[difference_std], 
					'|DIC|':[average_diff_abs], 'STD |DIC|':[difference_std_abs],
					'MSE':[MSE], 'R^2':[r_coeff]
						}

	# Statististics
	results_ordered_dict = OrderedDict()
	results_ordered_dict['DIC train'] = average_diff_train
	results_ordered_dict['STD DIC train'] = difference_std_train
	results_ordered_dict['|DIC| train'] = average_diff_abs_train
	results_ordered_dict['STD |DIC| train'] = difference_std_abs_train
	results_ordered_dict['MSE train'] = MSE_train
	results_ordered_dict['R^2 train'] = r_coeff_train
	results_ordered_dict['Agreement train'] = agreement_train
	results_ordered_dict['DIC'] = average_diff
	results_ordered_dict['STD DIC'] = difference_std
	results_ordered_dict['|DIC|'] = average_diff_abs
	results_ordered_dict['STD |DIC|'] = difference_std_abs
	results_ordered_dict['MSE'] = MSE
	results_ordered_dict['R^2'] = r_coeff
	results_ordered_dict['Agreement'] = agreement


	results_arr_equal = np.full_like(result_arr_train,0)
	results_arr_equal[0:len(y_test_all),0] = result_arr[:,0]
	results_arr_equal[0:len(y_test_all),1] = result_arr[:,1]
	results_arr_equal[0:len(y_test_all),2] = result_arr[:,2]
	results_arr_equal[0:len(y_test_all),4] = result_arr[:,4]


	results_arrays_dict = OrderedDict()
	results_arrays_dict['Training Image name'] = result_arr_train[:,2]
	results_arrays_dict['Training targets'] = result_arr_train[:,0]
	results_arrays_dict['Training predictions'] = result_arr_train[:,1]
	results_arrays_dict['Training agreement'] = result_arr_train[:,4]
	results_arrays_dict['Test Image name'] = results_arr_equal[:,2]
	results_arrays_dict['Test targets'] = results_arr_equal[:,0]
	results_arrays_dict['Test predictions'] = results_arr_equal[:,1]
	results_arrays_dict['Test agreement'] = results_arr_equal[:,4]

	results_arrays_dataframe = pd.DataFrame(results_arrays_dict, index=list(range(0,len(y_train_all))))
	excel_writer_one = pd.ExcelWriter(results_path+'/ResultsPredictions.xlsx', engine='xlsxwriter')
	results_arrays_dataframe.to_excel(excel_writer_one, sheet_name='Sheet1')
	excel_writer_one.save()

	results_dataframe = pd.DataFrame(results_ordered_dict, index=[0])
	excel_writer_two = pd.ExcelWriter(results_path+'/ResultsStats.xlsx', engine='xlsxwriter')
	results_dataframe.to_excel(excel_writer_two, sheet_name='Sheet1')
	excel_writer_two.save()

	#print('Plants used for training are', train_plants, 'for validation', validate_plants, 'for testing' , test_plants)
	# print(difference_arr)
	# print(result_arr)
	print('Average difference is', average_diff, 'Test MSE is ' ,MSE)

try:
	if split_decision == 'all':
		for current_split in split_all:
			current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
			results_path = ('./Results/CVPPP_split_'+ str(current_split)+' '+current_time)
			os.makedirs(results_path)
			split_load = splits[current_split]
			print('Split decision was '+str(current_split)+' Current split is '+str(split_load))
			current_data = get_data(split_load)
			x_train_all = current_data[0]
			x_val_all = current_data[1]
			x_test_all = current_data[2]
			y_train_all = current_data[3]
			y_val_all = current_data[4]
			y_test_all = current_data[5]
			x_train_set = current_data[6]
			x_val_set = current_data[7]
			x_test_set = current_data[8]
			x_train_img = current_data[9]
			x_val_img = current_data[10]
			x_test_img = current_data[11]
			traning_model = counter_model(x_train_all,x_val_all,y_train_all,y_val_all)
			testing_results(traning_model, x_train_all, x_test_all, y_train_all, y_test_all, x_train_img, x_test_img)
			del traning_model
			
	else:
		current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
		results_path = ('./Results/CVPPP_split '+split_decision+' '+current_time)
		os.makedirs(results_path)
		print('Split decision was '+split_decision+' Current split is '+str(split_load))
		current_data = get_data(split_load)
		x_train_all = current_data[0]
		x_val_all = current_data[1]
		x_test_all = current_data[2]
		y_train_all = current_data[3]
		y_val_all = current_data[4]
		y_test_all = current_data[5]
		x_train_set = current_data[6]
		x_val_set = current_data[7]
		x_test_set = current_data[8]
		x_train_img = current_data[9]
		x_val_img = current_data[10]
		x_test_img = current_data[11]
		traning_model = counter_model(x_train_all,x_val_all,y_train_all,y_val_all)
		testing_results(traning_model, x_train_all, x_test_all, y_train_all, y_test_all, x_train_img, x_test_img)
except Exception:
	traceback.print_exc()