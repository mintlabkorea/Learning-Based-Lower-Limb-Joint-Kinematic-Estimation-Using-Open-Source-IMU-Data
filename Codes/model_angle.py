# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 21:14:46 2023

@author: user
"""

import torch
from torch import nn

class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()
		self.lstm_size = 128
		self.in_size = 72
		self.out_size = 10
		self.num_layers = 3

		'''
		self.embedding = nn.Embedding(
			num_embeddings=n_vocab,
			embedding_dim=self.embedding_dim,
			)
		'''
		self.lstm = nn.LSTM(
			input_size=self.in_size,
			hidden_size=self.lstm_size,
			num_layers=self.num_layers,
			dropout=0.2,
			batch_first=True
		)
		self.fc = nn.Linear(self.lstm_size, self.out_size) 
		

	def forward(self, x, prev_state):
		output, state = self.lstm(x, prev_state)
		logits = self.fc(output)
		return logits, state

	def init_state(self, sequence_length):
		return (torch.zeros(self.num_layers, sequence_length, self.lstm_size),
		        torch.zeros(self.num_layers, sequence_length, self.lstm_size))