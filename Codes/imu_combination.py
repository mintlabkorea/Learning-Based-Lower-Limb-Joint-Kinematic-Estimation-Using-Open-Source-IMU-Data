import os, sys
from itertools import combinations

class combination:
	def __init__(self):
		self.imus = ['pelvis', 'tibia_r', 'femur_r', 'tibia_l', 'femur_l', 'calcn_r', 'calcn_l']
		
		self.list1 = list(combinations(self.imus, 1))
		self.list2 = list(combinations(self.imus, 2))
		self.list3 = list(combinations(self.imus, 3))
		self.list4 = list(combinations(self.imus, 4))
		self.list5 = list(combinations(self.imus, 5))
		self.list6 = list(combinations(self.imus, 6))
		self.list_combination = self.list1 + self.list2 + self.list3 + self.list4+ self.list5 + self.list6
		
		self.comb_index = dict(enumerate(self.list_combination, start=1))
		
	def combination_loader(self, x):
		imu_comb = self.comb_index[x]
		return imu_comb