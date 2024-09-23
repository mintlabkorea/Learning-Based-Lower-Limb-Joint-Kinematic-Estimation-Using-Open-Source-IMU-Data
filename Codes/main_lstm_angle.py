# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 20:54:44 2023

@author: user
"""

import os, sys
import importlib
from trainertoy_angle3_mint import trainer
from utils import error_logger
from plotter_angle import raw_data_plotter, result_plotter
#from model import Model
import torch

device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
	torch.set_default_tensor_type(torch.cuda.FloatTensor)


def main():
	if len(sys.argv) != 2:
		error_logger("Include parameters.")

	param_path = sys.argv[1]
	if not os.path.isfile(param_path):
		error_logger("Parameter file not found.")

	config = importlib.import_module(param_path[:-3]).PARAMS
	train = trainer(config)


	imu_data, kin_data, data_labels = train.load_mat()
	
	

	if config['plot_raw_data']:
		if config['add_shank_m_foot']:
			error_logger("Please turn off 'add_shank_m_foot' for plotting")
		raw_data_plotter(imu_data, kin_data, 
			config['xcriteria'], config['ycriteria'], 
			data_labels, 
			config['paths']['plot_path_rawdata'])
	
	
	# train_imu, test_imu, train_dyn, test_dyn, val_imu, val_dyn = train.dataprocessor(imu_data, kin_data)
	

	
	# model = train.valid(train_imu, train_dyn, val_imu, val_dyn)
	# model.to(device)
	
	
	# train.test_lr(test_imu, test_dyn, data_labels, model)
	train.load_model_and_test()
		
	if config['plot_result']:
		if not config['save_result']:
			error_logger("Please turn on 'save_result to plot results.'")
		result_plotter(config['paths']['result_path'], config['paths']['plot_path_results'], config['target_output'])
		
		
if __name__ == "__main__":
	main()