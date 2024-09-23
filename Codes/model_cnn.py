# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 21:14:46 2023

@author: user
"""

import torch
from torch import nn
import torch.nn.functional as F

class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		
		self.layer1 = nn.Sequential(
			nn.Conv1d(in_channels=72, out_channels=24, kernel_size=5, stride=1, padding=2),
			nn.ReLU(),
			#nn.MaxPool1d(kernel_size=2)
			)
		
		self.layer2 = nn.Sequential(
			nn.Conv1d(in_channels=24, out_channels=8, kernel_size=5, stride=1, padding=2),
			nn.ReLU(),
			#nn.MaxPool1d(kernel_size=2)
			)
		
		self.layer3 = nn.Sequential(
			nn.Conv1d(in_channels=8, out_channels=3, kernel_size=5, stride=1, padding=2),
			nn.ReLU(),
			#nn.MaxPool1d(kernel_size=2)
			)
		
		self.fc = nn.Linear(in_features=50, out_features=50)
		nn.init.xavier_uniform_(self.fc.weight)
		
	def forward(self,x):
		out1 = self.layer1(x)
		out2 = self.layer2(out1)
		out3 = self.layer3(out2)
		out = self.fc(out3)
		return out