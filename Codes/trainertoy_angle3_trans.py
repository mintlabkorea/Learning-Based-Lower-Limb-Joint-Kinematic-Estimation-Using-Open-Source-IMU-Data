# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 03:01:02 2023

@author: user
"""

import os
import h5py
import numpy as np
import copy
import random
from utils import flatten_list, error_logger, create_dir
import pickle
import torch
from torch import nn, optim
from model_angle import Model
device=torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
if torch.cuda.is_available():
	torch.set_default_tensor_type(torch.cuda.FloatTensor)
class trainer():
	def __init__(self, config):
		self.datapath = config['paths']['data_path']
		self.weights_path = config['paths']['weight_path']
		self.result_path = config['paths']['result_path']

		self.cur_data = [os.path.join(self.datapath, x) for x in os.listdir(self.datapath)]
		self.xcriteria = config['xcriteria']
		self.ycriteria = config['ycriteria']
		
		self.add_shank_m_foot = config['add_shank_m_foot']
		self.manual_apgrf_flip = config['manual_apgrf_flip']
		self.target_output = config['target_output']
		self.data_major_div = config['data_major_div']
		self.data_minor_add = config['data_minor_add']

		self.target_output = config['target_output']
		self.Trained_model = config['Trained_model']
		self.save_weight = config['save_weight']
		self.save_result = config['save_result']
		self.mode = 0
		#self.ic = imu_comb
		#self.icstr = "_".join(self.ic)
	

	def training_lr(self, train_imu_ten, train_dyn_ten, num, model):
		#from sklearn.linear_model import LinearRegression
		model.train()
		self.mode = self.mode_indexing()
		
		optimizer = optim.Adam(model.parameters(), lr=0.001)
		
		criterion = nn.MSELoss()
		
		for epoch in range(int(num)):
			state_h, state_c = model.init_state(train_imu_ten.shape[0])
			
			
			optimizer.zero_grad()

			pred, (state_h, state_c) = model(train_imu_ten, (state_h, state_c))
			#pred = model(train_imu_ten)
			#print(pred.shape)
			#print(train_dyn_ten[:, self.mode, :].shape)
			loss = criterion(pred.transpose(1,2), train_dyn_ten[:, self.mode, :])
			#loss = criterion(pred, train_dyn_ten[:, self.mode, :])
			
			state_h = state_h.detach()
			state_c = state_c.detach()

			loss.backward()
			optimizer.step()
			print({ 'epoch': epoch, 'loss': loss.item() })
	
	def valid(self, train_imu_ten, train_dyn_ten, val_imu_ten, val_dyn_ten):
		if self.Trained_model == False:
			epoch_start = 500
			epoch_step = 500
			cnt = 0
			
	
			model=Model()
			model.to(device)
			self.training_lr(train_imu_ten, train_dyn_ten, epoch_start, model)
			
			model.eval()
			state_h, state_c = model.init_state(val_imu_ten.shape[0])
			
			
			v_predicted, (state_h, state_c) = model(val_imu_ten, (state_h, state_c))
			#v_predicted = model(val_imu_ten)
			v_predictedt = v_predicted.transpose(1,2)
			#v_predictedt = v_predicted
			
			error_val1 = 0
			error_val2 = 0
			error_val3 = 0
# 			error_val4 = 0
# 			error_val5 = 0
# 			error_val6 = 0
# 			error_val7 = 0
# 			error_val8 = 0
# 			error_val9 = 0
# 			error_val10 = 0
			
			for i_batch in range(v_predictedt.shape[0]):
				v_pred = v_predictedt[i_batch, :, :]
				v_act = val_dyn_ten[i_batch, self.mode, :]
				#print(v_pred.shape)
				#print(v_act.shape)
				#v_preds=v_pred.squeeze()
				error_val1 += torch.sum((v_pred[0, :] - v_act[0, :])**2)
				error_val2 += torch.sum((v_pred[1, :] - v_act[1, :])**2)
				error_val3 += torch.sum((v_pred[2, :] - v_act[2, :])**2)
# 				error_val4 += torch.sum((v_pred[3, :] - v_act[3, :])**2)
# 				error_val5 += torch.sum((v_pred[4, :] - v_act[4, :])**2)
# 				error_val6 += torch.sum((v_pred[5, :] - v_act[5, :])**2)
# 				error_val7 += torch.sum((v_pred[6, :] - v_act[6, :])**2)
# 				error_val8 += torch.sum((v_pred[7, :] - v_act[7, :])**2)
# 				error_val9 += torch.sum((v_pred[8, :] - v_act[8, :])**2)
# 				error_val10 += torch.sum((v_pred[9, :] - v_act[9, :])**2)
			rms_error1 = (error_val1/((i_batch+1)*v_pred.shape[1]))**0.5
			rms_error2 = (error_val2/((i_batch+1)*v_pred.shape[1]))**0.5
			rms_error3 = (error_val3/((i_batch+1)*v_pred.shape[1]))**0.5
# 			rms_error4 = (error_val4/((i_batch+1)*v_pred.shape[1]))**0.5
# 			rms_error5 = (error_val5/((i_batch+1)*v_pred.shape[1]))**0.5
# 			rms_error6 = (error_val6/((i_batch+1)*v_pred.shape[1]))**0.5
# 			rms_error7 = (error_val7/((i_batch+1)*v_pred.shape[1]))**0.5
# 			rms_error8 = (error_val8/((i_batch+1)*v_pred.shape[1]))**0.5
# 			rms_error9 = (error_val9/((i_batch+1)*v_pred.shape[1]))**0.5
# 			rms_error10 = (error_val10/((i_batch+1)*v_pred.shape[1]))**0.5
			print({'epoch' : epoch_start})
			print("Mean Squared Error 1: {:.4f}".format(rms_error1))
			print("Mean Squared Error 2: {:.4f}".format(rms_error2))
			print("Mean Squared Error 3: {:.4f}".format(rms_error3))
			rms_error = rms_error1 + rms_error2 + rms_error3 #+ rms_error4 + rms_error5 + rms_error6 + rms_error7 + rms_error8 + rms_error9 + rms_error10
			loss_val1 = rms_error.cpu().detach().numpy()
			
			
			while(1):
				cnt += 1
				model_copy = copy.deepcopy(model)
				self.training_lr(train_imu_ten, train_dyn_ten, epoch_step, model)
				
				model.eval()
				state_h, state_c = model.init_state(val_imu_ten.shape[0])
				
				
				v_predicted, (state_h, state_c) = model(val_imu_ten, (state_h, state_c))
				#v_predicted = model(val_imu_ten)
				v_predictedt = v_predicted.transpose(1,2)
				#v_predictedt = v_predicted
				
				error_val1 = 0
				error_val2 = 0
				error_val3 = 0
# 				error_val4 = 0
# 				error_val5 = 0
# 				error_val6 = 0
# 				error_val7 = 0
# 				error_val8 = 0
# 				error_val9 = 0
# 				error_val10 = 0
				
				for i_batch in range(v_predictedt.shape[0]):
					v_pred = v_predictedt[i_batch, :, :]
					v_act = val_dyn_ten[i_batch, self.mode, :]
					#print(v_pred.shape)
					#print(v_act.shape)
					#v_preds=v_pred.squeeze()
					error_val1 += torch.sum((v_pred[0, :] - v_act[0, :])**2)
					error_val2 += torch.sum((v_pred[1, :] - v_act[1, :])**2)
					error_val3 += torch.sum((v_pred[2, :] - v_act[2, :])**2)
# 					error_val4 += torch.sum((v_pred[3, :] - v_act[3, :])**2)
# 					error_val5 += torch.sum((v_pred[4, :] - v_act[4, :])**2)
# 					error_val6 += torch.sum((v_pred[5, :] - v_act[5, :])**2)
# 					error_val7 += torch.sum((v_pred[6, :] - v_act[6, :])**2)
# 					error_val8 += torch.sum((v_pred[7, :] - v_act[7, :])**2)
# 					error_val9 += torch.sum((v_pred[8, :] - v_act[8, :])**2)
# 					error_val10 += torch.sum((v_pred[9, :] - v_act[9, :])**2)
				rms_error1 = (error_val1/((i_batch+1)*v_pred.shape[1]))**0.5
				rms_error2 = (error_val2/((i_batch+1)*v_pred.shape[1]))**0.5
				rms_error3 = (error_val3/((i_batch+1)*v_pred.shape[1]))**0.5
# 				rms_error4 = (error_val4/((i_batch+1)*v_pred.shape[1]))**0.5
# 				rms_error5 = (error_val5/((i_batch+1)*v_pred.shape[1]))**0.5
# 				rms_error6 = (error_val6/((i_batch+1)*v_pred.shape[1]))**0.5
# 				rms_error7 = (error_val7/((i_batch+1)*v_pred.shape[1]))**0.5
# 				rms_error8 = (error_val8/((i_batch+1)*v_pred.shape[1]))**0.5
# 				rms_error9 = (error_val9/((i_batch+1)*v_pred.shape[1]))**0.5
# 				rms_error10 = (error_val10/((i_batch+1)*v_pred.shape[1]))**0.5
				print({'epoch' : epoch_start + epoch_step*cnt})
				print("Mean Squared Error 1: {:.4f}".format(rms_error1))
				print("Mean Squared Error 2: {:.4f}".format(rms_error2))
				print("Mean Squared Error 3: {:.4f}".format(rms_error3))
				rms_error = rms_error1 + rms_error2 + rms_error3 #+ rms_error4 + rms_error5 + rms_error6 + rms_error7 + rms_error8 + rms_error9 + rms_error10
				loss_val2 = rms_error.cpu().detach().numpy()
				
				if loss_val1 > loss_val2:
					if epoch_start + epoch_step*cnt >= 3000:
						model_final = copy.deepcopy(model)
						print({'final epoch' : epoch_start + epoch_step*cnt})
						break
					else:
						loss_val1 = loss_val2
						print("next epoch")
				elif loss_val1 < loss_val2:
					model_final = copy.deepcopy(model_copy)
					print({'final epoch' : epoch_start + epoch_step*(cnt-1)})
					break
		elif self.Trained_model == True:
			model_final = torch.load('./weights/generalized_' + self.target_output + '.pt')


		return model_final

	
	def transfer(self, train_imu_ten, train_dyn_ten, val_imu_ten, val_dyn_ten, model, est, esp, emx):
		epoch_start = est
		epoch_step = esp
		cnt = 0
		

		self.training_lr(train_imu_ten, train_dyn_ten, epoch_start, model)
		
		model.eval()
		state_h, state_c = model.init_state(val_imu_ten.shape[0])
		
		
		v_predicted, (state_h, state_c) = model(val_imu_ten, (state_h, state_c))
		v_predictedt = v_predicted.transpose(1,2)
		#v_predictedt = model(val_imu_ten)
		
		error_val1 = 0
		error_val2 = 0
		error_val3 = 0
# 		error_val4 = 0
# 		error_val5 = 0
# 		error_val6 = 0
# 		error_val7 = 0
# 		error_val8 = 0
# 		error_val9 = 0
# 		error_val10 = 0
		
		for i_batch in range(v_predictedt.shape[0]):
			v_pred = v_predictedt[i_batch, :, :]
			v_act = val_dyn_ten[i_batch, self.mode, :]
			#print(v_pred.shape)
			#print(v_act.shape)
			#v_preds=v_pred.squeeze()
			error_val1 += torch.sum((v_pred[0, :] - v_act[0, :])**2)
			error_val2 += torch.sum((v_pred[1, :] - v_act[1, :])**2)
			error_val3 += torch.sum((v_pred[2, :] - v_act[2, :])**2)
# 			error_val4 += torch.sum((v_pred[3, :] - v_act[3, :])**2)
# 			error_val5 += torch.sum((v_pred[4, :] - v_act[4, :])**2)
# 			error_val6 += torch.sum((v_pred[5, :] - v_act[5, :])**2)
# 			error_val7 += torch.sum((v_pred[6, :] - v_act[6, :])**2)
# 			error_val8 += torch.sum((v_pred[7, :] - v_act[7, :])**2)
# 			error_val9 += torch.sum((v_pred[8, :] - v_act[8, :])**2)
# 			error_val10 += torch.sum((v_pred[9, :] - v_act[9, :])**2)
		rms_error1 = (error_val1/((i_batch+1)*v_pred.shape[1]))**0.5
		rms_error2 = (error_val2/((i_batch+1)*v_pred.shape[1]))**0.5
		rms_error3 = (error_val3/((i_batch+1)*v_pred.shape[1]))**0.5
# 		rms_error4 = (error_val4/((i_batch+1)*v_pred.shape[1]))**0.5
# 		rms_error5 = (error_val5/((i_batch+1)*v_pred.shape[1]))**0.5
# 		rms_error6 = (error_val6/((i_batch+1)*v_pred.shape[1]))**0.5
# 		rms_error7 = (error_val7/((i_batch+1)*v_pred.shape[1]))**0.5
# 		rms_error8 = (error_val8/((i_batch+1)*v_pred.shape[1]))**0.5
# 		rms_error9 = (error_val9/((i_batch+1)*v_pred.shape[1]))**0.5
# 		rms_error10 = (error_val10/((i_batch+1)*v_pred.shape[1]))**0.5
		print({'epoch' : epoch_start})
# 		print("Mean Squared Error 1: {:.4f}".format(rms_error1))
# 		print("Mean Squared Error 2: {:.4f}".format(rms_error2))
# 		print("Mean Squared Error 3: {:.4f}".format(rms_error3))
		rms_error = rms_error1 + rms_error2 + rms_error3# + rms_error4 + rms_error5 + rms_error6 + rms_error7 + rms_error8 + rms_error9 + rms_error10
		loss_val1 = rms_error.cpu().detach().numpy()
		
		
		while(1):
			cnt += 1
			model_copy = copy.deepcopy(model)
			self.training_lr(train_imu_ten, train_dyn_ten, epoch_step, model)
			
			model.eval()
			state_h, state_c = model.init_state(val_imu_ten.shape[0])
			
			
			v_predicted, (state_h, state_c) = model(val_imu_ten, (state_h, state_c))
			v_predictedt = v_predicted.transpose(1,2)
			#v_predictedt = model(val_imu_ten)
			
			error_val1 = 0
			error_val2 = 0
			error_val3 = 0
# 			error_val4 = 0
# 			error_val5 = 0
# 			error_val6 = 0
# 			error_val7 = 0
# 			error_val8 = 0
# 			error_val9 = 0
# 			error_val10 = 0
			
			for i_batch in range(v_predictedt.shape[0]):
				v_pred = v_predictedt[i_batch, :, :]
				v_act = val_dyn_ten[i_batch, self.mode, :]
				#print(v_pred.shape)
				#print(v_act.shape)
				#v_preds=v_pred.squeeze()
				error_val1 += torch.sum((v_pred[0, :] - v_act[0, :])**2)
				error_val2 += torch.sum((v_pred[1, :] - v_act[1, :])**2)
				error_val3 += torch.sum((v_pred[2, :] - v_act[2, :])**2)
# 				error_val4 += torch.sum((v_pred[3, :] - v_act[3, :])**2)
# 				error_val5 += torch.sum((v_pred[4, :] - v_act[4, :])**2)
# 				error_val6 += torch.sum((v_pred[5, :] - v_act[5, :])**2)
# 				error_val7 += torch.sum((v_pred[6, :] - v_act[6, :])**2)
# 				error_val8 += torch.sum((v_pred[7, :] - v_act[7, :])**2)
# 				error_val9 += torch.sum((v_pred[8, :] - v_act[8, :])**2)
# 				error_val10 += torch.sum((v_pred[9, :] - v_act[9, :])**2)
			rms_error1 = (error_val1/((i_batch+1)*v_pred.shape[1]))**0.5
			rms_error2 = (error_val2/((i_batch+1)*v_pred.shape[1]))**0.5
			rms_error3 = (error_val3/((i_batch+1)*v_pred.shape[1]))**0.5
# 			rms_error4 = (error_val4/((i_batch+1)*v_pred.shape[1]))**0.5
# 			rms_error5 = (error_val5/((i_batch+1)*v_pred.shape[1]))**0.5
# 			rms_error6 = (error_val6/((i_batch+1)*v_pred.shape[1]))**0.5
# 			rms_error7 = (error_val7/((i_batch+1)*v_pred.shape[1]))**0.5
# 			rms_error8 = (error_val8/((i_batch+1)*v_pred.shape[1]))**0.5
# 			rms_error9 = (error_val9/((i_batch+1)*v_pred.shape[1]))**0.5
# 			rms_error10 = (error_val10/((i_batch+1)*v_pred.shape[1]))**0.5
			print({'epoch' : epoch_start + epoch_step*cnt})
			print("Mean Squared Error 1: {:.4f}".format(rms_error1))
			print("Mean Squared Error 2: {:.4f}".format(rms_error2))
			print("Mean Squared Error 3: {:.4f}".format(rms_error3))
			rms_error = rms_error1 + rms_error2 + rms_error3# + rms_error4 + rms_error5 + rms_error6 + rms_error7 + rms_error8 + rms_error9 + rms_error10
			loss_val2 = rms_error.cpu().detach().numpy()
			
			if loss_val1 > loss_val2:
				if epoch_start + epoch_step*cnt >= emx:
					model_final = copy.deepcopy(model)
					print({'final epoch' : epoch_start + epoch_step*cnt})
					break
				else:
					loss_val1 = loss_val2
					print("next epoch")
			elif loss_val1 < loss_val2:
				model_final = copy.deepcopy(model_copy)
				print({'final epoch' : epoch_start + epoch_step*(cnt-1)})
				break
		
		#weight_name = os.path.join('./weights', self.weights_path + '_trans_' + self.icstr + '.pt')
		#torch.save(model_final, weight_name)

		return model_final
	
	def test_lr(self, test_imu_ten, test_dyn_ten, data_labels, model):
		#from sklearn.metrics import r2_score
		
		
		all_val = [x for x in range(test_imu_ten.shape[0])]
		#test_val = [x*self.data_major_div + self.data_minor_add for x in range(int(imu_data.shape[0]*.2))]
		trainf_val = [x for x in range(int(0.8*test_imu_ten.shape[0]))]

		test_val = [x for x in all_val if x not in trainf_val]
		#val_val = [x*self.data_major_div + self.data_minor_add for x in range(int(len(trainf_val)*.2))]
		val_val = random.sample(trainf_val, k=int(len(trainf_val)*.5))
		train_val = [x for x in all_val if x not in test_val and x not in val_val]
		test_train_imu_ten = test_imu_ten[train_val, :]
		test_test_imu_ten = test_imu_ten[test_val, :]
		test_val_imu_ten = test_imu_ten[val_val, :]
		test_train_dyn_ten = test_dyn_ten[train_val, :]
		test_test_dyn_ten = test_dyn_ten[test_val, :]
		test_val_dyn_ten = test_dyn_ten[val_val, :]
		#test_train_imu_ten=torch.Tensor(test_train_imu)
		#test_test_imu_ten=torch.Tensor(test_test_imu)
		#test_val_imu_ten=torch.Tensor(test_val_imu)
		#test_train_dyn_ten=torch.Tensor(test_train_dyn)
		#test_test_dyn_ten=torch.Tensor(test_test_dyn)
		#test_val_dyn_ten=torch.Tensor(test_val_dyn)
		
		model_final = self.transfer(test_train_imu_ten, test_train_dyn_ten, test_val_imu_ten, test_val_dyn_ten, model, 1, 1, 15)
		
		
		model_final.eval()
		
		state_h, state_c = model_final.init_state(test_test_imu_ten.shape[0])
		state_h.to(device), state_c.to(device)
		
		y_predicted, (state_h, state_c) = model_final(test_test_imu_ten, (state_h, state_c))
		y_predictedt = y_predicted.transpose(1,2)
		#y_predictedt = model_final(test_test_imu_ten)
		
		error_val1 = 0
		error_val2 = 0
		error_val3 = 0
# 		error_val4 = 0
# 		error_val5 = 0
# 		error_val6 = 0
# 		error_val7 = 0
# 		error_val8 = 0
# 		error_val9 = 0
# 		error_val10 = 0
		
		for i_batch in range(y_predictedt.shape[0]):
			y_pred = y_predictedt[i_batch, :, :]
			y_act = test_test_dyn_ten[i_batch, self.mode, :]
			#y_preds=y_pred.squeeze()
			y_predn=y_pred.cpu().detach().numpy()
			y_actn=y_act.cpu().detach().numpy()
			error_val1 += torch.sum((y_pred[0, :] - y_act[0, :])**2)
			error_val2 += torch.sum((y_pred[1, :] - y_act[1, :])**2)
			error_val3 += torch.sum((y_pred[2, :] - y_act[2, :])**2)
# 			error_val4 += torch.sum((y_pred[3, :] - y_act[3, :])**2)
# 			error_val5 += torch.sum((y_pred[4, :] - y_act[4, :])**2)
# 			error_val6 += torch.sum((y_pred[5, :] - y_act[5, :])**2)
# 			error_val7 += torch.sum((y_pred[6, :] - y_act[6, :])**2)
# 			error_val8 += torch.sum((y_pred[7, :] - y_act[7, :])**2)
# 			error_val9 += torch.sum((y_pred[8, :] - y_act[8, :])**2)
# 			error_val10 += torch.sum((y_pred[9, :] - y_act[9, :])**2)
			rms_error1 = (error_val1/((i_batch+1)*y_pred.shape[1]))**0.5
			rms_error2 = (error_val2/((i_batch+1)*y_pred.shape[1]))**0.5
			rms_error3 = (error_val3/((i_batch+1)*y_pred.shape[1]))**0.5
# 			rms_error4 = (error_val4/((i_batch+1)*y_pred.shape[1]))**0.5
# 			rms_error5 = (error_val5/((i_batch+1)*y_pred.shape[1]))**0.5
# 			rms_error6 = (error_val6/((i_batch+1)*y_pred.shape[1]))**0.5
# 			rms_error7 = (error_val7/((i_batch+1)*y_pred.shape[1]))**0.5
# 			rms_error8 = (error_val8/((i_batch+1)*y_pred.shape[1]))**0.5
# 			rms_error9 = (error_val9/((i_batch+1)*y_pred.shape[1]))**0.5
# 			rms_error10 = (error_val10/((i_batch+1)*y_pred.shape[1]))**0.5
			#print("Mean Squared Error 1: {:.4f}".format(rms_error1))
			#print("Mean Squared Error 2: {:.4f}".format(rms_error2))
			#print("Mean Squared Error 3: {:.4f}".format(rms_error3))
			rms_error1n = rms_error1.cpu().detach().numpy()
			rms_error2n = rms_error2.cpu().detach().numpy()
			rms_error3n = rms_error3.cpu().detach().numpy()
# 			rms_error4n = rms_error4.cpu().detach().numpy()
# 			rms_error5n = rms_error5.cpu().detach().numpy()
# 			rms_error6n = rms_error6.cpu().detach().numpy()
# 			rms_error7n = rms_error7.cpu().detach().numpy()
# 			rms_error8n = rms_error8.cpu().detach().numpy()
# 			rms_error9n = rms_error9.cpu().detach().numpy()
# 			rms_error10n = rms_error10.cpu().detach().numpy()
			rms_error = [rms_error1n, rms_error2n, rms_error3n]#, rms_error4n, rms_error5n, rms_error6n, rms_error7n, rms_error8n, rms_error9n, rms_error10n]
			rms_errorn = np.array(rms_error)


			
			if self.save_result:
				fname1 = self.datapath[2:] + '_' + data_labels[i_batch] + '_' + self.target_output + '_1_' + 'twodir.csv'
				fname2 = self.datapath[2:] + '_' + data_labels[i_batch] + '_' + self.target_output + '_2_'  + 'twodir.csv'
				fname3 = self.datapath[2:] + '_' + data_labels[i_batch] + '_' + self.target_output + '_3_'  + 'twodir.csv'
				fnamee = self.datapath[2:] + '_' + data_labels[i_batch] + '_' + self.target_output + '_rmse_' + 'twodir.csv'
				create_dir(os.path.join('results', self.result_path, self.target_output))
				fpath1 = os.path.join('results', self.result_path, self.target_output, fname1)
				fpath2 = os.path.join('results', self.result_path, self.target_output, fname2)
				fpath3 = os.path.join('results', self.result_path, self.target_output, fname3)
				fpathe = os.path.join('results', self.result_path, self.target_output, fnamee)

				np.savetxt(fpath1, np.concatenate((y_predn[0].reshape(-1, 1), y_actn[0].reshape(-1, 1)), axis =1), delimiter=',')
				np.savetxt(fpath2, np.concatenate((y_predn[1].reshape(-1, 1), y_actn[1].reshape(-1, 1)), axis =1), delimiter=',')
				np.savetxt(fpath3, np.concatenate((y_predn[2].reshape(-1, 1), y_actn[2].reshape(-1, 1)), axis =1), delimiter=',')
				np.savetxt(fpathe, rms_errorn)




	def mode_indexing(self):
		if self.target_output == 'pelvis':
			return [0, 1, 2]
		elif self.target_output == 'pelvis_t':
			return [3, 4, 5]
		elif self.target_output == 'hip_r':
			return [0, 1, 2]
		elif self.target_output == 'knee_ankle_r':
			return [3, 4, 5]
		elif self.target_output == 'hip_l':
			return [6, 7, 8]
		elif self.target_output == 'knee_ankle_l':
			return [9, 10, 11]
		elif self.target_output == 'lumbar':
			return [18, 19 ,20]
		elif self.target_output == 'angles':
			return [6, 7, 8, 9, 11, 12, 13, 14, 15, 17]
		else:
			error_logger("Wrong target mode. Current mode: {}".format(self.target_output))

	
	def dataprocessor(self, imu_data, kin_data):
		tmp_train_imu = imu_data[0:5]
		tmp_test_imu = imu_data[7:8]
		tmp_val_imu = imu_data[5:6]
		tmp_train_dyn = kin_data[0:5]
		tmp_test_dyn = kin_data[7:8]
		tmp_val_dyn = kin_data[5:6]
		
		train_imu = []
		train_dyn = []
		val_imu = []
		val_dyn = []
		test_imu = []
		test_dyn = []

		for i in range(len(tmp_train_imu)):
			train_imu.append(tmp_train_imu[i])
			train_dyn.append(tmp_train_dyn[i])
		for i in range(len(tmp_val_imu)):
			val_imu.append(tmp_val_imu[i])
			val_dyn.append(tmp_val_dyn[i])
		for i in range(len(tmp_test_imu)):
			test_imu.append(tmp_test_imu[i])
			test_dyn.append(tmp_test_dyn[i])
		
		test_imu = np.concatenate(np.array(test_imu, dtype=object), 0)
		test_dyn = np.concatenate(np.array(test_dyn, dtype=object), 0)
		train_imu = np.concatenate(np.array(train_imu, dtype=object), 0)
		train_dyn = np.concatenate(np.array(train_dyn, dtype=object), 0)
		val_imu = np.concatenate(np.array(val_imu, dtype=object), 0)
		val_dyn = np.concatenate(np.array(val_dyn, dtype=object), 0)
		
		test_imu = np.squeeze(test_imu)
		test_dyn = np.squeeze(test_dyn)
		train_imu = np.squeeze(train_imu)
		train_dyn = np.squeeze(train_dyn)
		val_imu = np.squeeze(val_imu)
		val_dyn = np.squeeze(val_dyn)
		
		test_imu = test_imu.astype(float)
		test_dyn = test_dyn.astype(float)
		train_imu = train_imu.astype(float)
		train_dyn = train_dyn.astype(float)
		val_imu = val_imu.astype(float)
		val_dyn = val_dyn.astype(float)
		
		
		train_imu_tent=torch.Tensor(train_imu)
		test_imu_tent=torch.Tensor(test_imu)
		val_imu_tent=torch.Tensor(val_imu)
		train_imu_ten=train_imu_tent.transpose(1,2)
		test_imu_ten=test_imu_tent.transpose(1,2)
		val_imu_ten=val_imu_tent.transpose(1,2)
		train_dyn_ten=torch.Tensor(train_dyn)
		test_dyn_ten=torch.Tensor(test_dyn)
		val_dyn_ten=torch.Tensor(val_dyn)
		
		#print(train_imu_ten.shape)
		#print(test_imu_ten.shape)

		return train_imu_ten, test_imu_ten, train_dyn_ten, test_dyn_ten, val_imu_ten, val_dyn_ten

	def load_mat(self):
		imu_data = []
		kin_data = []
		data_labels = []

		for cur_mat in self.cur_data:
			if not cur_mat.endswith("mat"):
				continue
			if not "normalized" in cur_mat:
				continue
			if cur_mat.endswith("05.mat") or cur_mat.endswith("10.mat") or cur_mat.endswith("09.mat") or cur_mat.endswith("11.mat"):
				continue
			print(cur_mat)
			with h5py.File(cur_mat, 'r') as f:
				cur_imu = []
				cur_kin = []
				for cur_trial in f['normalized_data'].keys():

					print(cur_trial)
					
					
					if cur_trial == 'imu':
						for cur_f in f['normalized_data'][cur_trial].keys():
						
							for xcriterium in self.xcriteria:
								cur_imu.append(f['normalized_data'][cur_trial][cur_f][xcriterium][()])

					elif cur_trial == 'angle':
						for ycriterium in self.ycriteria:
							cur_kin.append(f['normalized_data'][cur_trial][ycriterium][()])
					
					else:
						error_logger("Wrong data.")

				#cur_imu = np.swapaxes(np.array(cur_imu), 0, 1)
				#cur_kin = np.swapaxes(np.array(cur_kin), 0, 1)
				cur_imu = np.array(cur_imu)
				cur_kin = np.array(cur_kin)

				print(cur_imu.shape)
				print(cur_kin.shape)
				
				cur_imu_sw, cur_kin_sw = self.sliding_window(cur_imu, cur_kin)
				
				trial_names = [cur_trial + '_' + str(x) for x in range(cur_imu_sw.shape[0])]
				data_labels.append(trial_names)
					
				print(cur_imu_sw.shape)
				print(cur_kin_sw.shape)
					
				imu_data.append(cur_imu_sw)
				kin_data.append(cur_kin_sw)

		#imu_data = np.concatenate(np.array(imu_data, dtype=object), 0)
		#kin_data = np.concatenate(np.array(kin_data, dtype=object), 0)
		#imu_data = np.array(imu_data)
		#kin_data = np.array(kin_data)
		#print(imu_data.shape)
		#print(kin_data.shape)
		#imu_data = np.squeeze(imu_data)
		#kin_data = np.squeeze(kin_data)
		#print(imu_data.shape)
		#print(kin_data.shape)
		
		return imu_data, kin_data, flatten_list(data_labels)
	
	def sliding_window(self, cur_imu, cur_kin):
		data_size = 50 #변경 가능
		overlapping_size = 25 #변경 가능
		data_num = (cur_imu.shape[2] - data_size)/(data_size - overlapping_size) + 1
		cur_imu_sw = []
		cur_kin_sw = []
		for cur_idx in range(int(data_num)):
			cur_imu_sw.append(cur_imu[:, :, (data_size - overlapping_size)*cur_idx : (data_size - overlapping_size)*cur_idx + data_size])
			cur_kin_sw.append(cur_kin[:, :, (data_size - overlapping_size)*cur_idx : (data_size - overlapping_size)*cur_idx + data_size])
		cur_imu_sw = np.array(cur_imu_sw)
		cur_kin_sw = np.array(cur_kin_sw)
		return cur_imu_sw, cur_kin_sw




