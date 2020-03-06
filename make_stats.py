from __future__ import division, print_function, absolute_import

import os
import traceback
import keras
import keras.backend as K
from keras.models import Sequential, Model, load_model
from sklearn.metrics import r2_score, mean_squared_error
from datetime import datetime
from collections import OrderedDict
from loading_data_CVPPP_all import get_data
import scipy.misc as misc
import matplotlib.pyplot as plt
import numpy as np
import h5py
import glob
import pandas as pd
import random
from PIL import Image, ImageOps


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




def testing_results(model, x_val_all, x_test_all, y_val_all, y_test_all, x_val_img, x_test_img):
	# valing set results
	predictions_val = model.predict(x_val_all)
	predictions_round_val = np.round(predictions_val)
	y_val_arr = np.reshape(y_val_all, (len(y_val_all),1))
	x_val_img_arr = np.reshape(x_val_img, (len(y_val_all),1))
	result_arr_val = np.concatenate((y_val_arr , predictions_round_val, x_val_img_arr, predictions_val), axis=1)

	difference_arr_val = np.array(np.round([(result_arr_val[h,1]-result_arr_val[h,0]) for h in range(0,len(y_val_arr))]))
	difference_arr_val = difference_arr_val.reshape(difference_arr_val.size, 1)
	difference_std_val = np.std(difference_arr_val)
	average_diff_val = np.average(difference_arr_val)

	difference_arr_abs_val = np.array(np.round([abs(result_arr_val[h,1]-result_arr_val[h,0]) for h in range(0,len(y_val_arr))]))
	difference_arr_abs_val = difference_arr_abs_val.reshape(difference_arr_abs_val.size, 1)
	difference_std_abs_val = np.std(difference_arr_abs_val)
	average_diff_abs_val = np.average(difference_arr_abs_val)

	prediction_equal_val = np.equal(result_arr_val[:,0],result_arr_val[:,1])
	prediction_equal_val = prediction_equal_val.astype(int)
	prediction_equal_val = np.reshape(prediction_equal_val, (len(result_arr_val),1))

	result_arr_val = np.concatenate((result_arr_val,prediction_equal_val), axis=1)

	mean_arr_val = np.mean(result_arr_val[:,1], axis=0)
	r_coeff_val = r2_score(y_val_arr, predictions_round_val)
	MSE_val = mean_squared_error(y_val_arr,predictions_round_val)
	agreement_sum_val = np.sum(prediction_equal_val)
	agreement_val = agreement_sum_val/len(y_val_all)

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
	results_dict = {'DIC val':[average_diff_val], 'STD DIC val':[difference_std_val], 
					'|DIC| val':[average_diff_abs_val], 'STD |DIC| val':[difference_std_abs_val],
					'MSE val':[MSE_val], 'R^2 val':[r_coeff_val],
					'DIC':[average_diff], 'STD DIC':[difference_std], 
					'|DIC|':[average_diff_abs], 'STD |DIC|':[difference_std_abs],
					'MSE':[MSE], 'R^2':[r_coeff]
						}

	# Statististics
	results_ordered_dict = OrderedDict()
	results_ordered_dict['DIC val'] = average_diff_val
	results_ordered_dict['STD DIC val'] = difference_std_val
	results_ordered_dict['|DIC| val'] = average_diff_abs_val
	results_ordered_dict['STD |DIC| val'] = difference_std_abs_val
	results_ordered_dict['MSE val'] = MSE_val
	results_ordered_dict['R^2 val'] = r_coeff_val
	results_ordered_dict['Agreement val'] = agreement_val
	results_ordered_dict['DIC'] = average_diff
	results_ordered_dict['STD DIC'] = difference_std
	results_ordered_dict['|DIC|'] = average_diff_abs
	results_ordered_dict['STD |DIC|'] = difference_std_abs
	results_ordered_dict['MSE'] = MSE
	results_ordered_dict['R^2'] = r_coeff
	results_ordered_dict['Agreement'] = agreement


	results_arr_equal = np.full_like(result_arr_val,0)
	results_arr_equal[0:len(y_test_all),0] = result_arr[:,0]
	results_arr_equal[0:len(y_test_all),1] = result_arr[:,1]
	results_arr_equal[0:len(y_test_all),2] = result_arr[:,2]
	results_arr_equal[0:len(y_test_all),3] = result_arr[:,3]
	results_arr_equal[0:len(y_test_all),4] = result_arr[:,4]


	results_arrays_dict = OrderedDict()
	results_arrays_dict['Val Image name'] = result_arr_val[:,2]
	results_arrays_dict['Val targets'] = result_arr_val[:,0]
	results_arrays_dict['Val round predictions'] = result_arr_val[:,1]
	results_arrays_dict['Val predictions'] = result_arr_val[:,3]
	results_arrays_dict['Val agreement'] = result_arr_val[:,4]
	results_arrays_dict['Test Image name'] = results_arr_equal[:,2]
	results_arrays_dict['Test targets'] = results_arr_equal[:,0]
	results_arrays_dict['Test round predictions'] = results_arr_equal[:,1]
	results_arrays_dict['Test predictions'] = results_arr_equal[:,3]
	results_arrays_dict['Test agreement'] = results_arr_equal[:,4]

	results_arrays_dataframe = pd.DataFrame(results_arrays_dict, index=list(range(0,len(y_val_all))))
	excel_writer_one = pd.ExcelWriter(results_path+'/Predictions.xlsx', engine='xlsxwriter')
	results_arrays_dataframe.to_excel(excel_writer_one, sheet_name='Sheet1')
	excel_writer_one.save()

	results_dataframe = pd.DataFrame(results_ordered_dict, index=[0])
	excel_writer_two = pd.ExcelWriter(results_path+'/Stats.xlsx', engine='xlsxwriter')
	results_dataframe.to_excel(excel_writer_two, sheet_name='Sheet1')
	excel_writer_two.save()

	#print('Plants used for valing are', val_plants, 'for validation', validate_plants, 'for testing' , test_plants)
	# print(difference_arr)
	# print(result_arr)
	print('Average difference is', average_diff, 'Test MSE is ' ,MSE)

def load_counter_model():

	model = load_model('./Results/CVPPP_split_2 2017-06-17 04_56_43/the_model.h5')

	return model

results_path = ('./Results/CVPPP_split_2_2')
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
traning_model = load_counter_model()
testing_results(traning_model, x_val_all, x_test_all, y_val_all, y_test_all, x_val_img, x_test_img)
