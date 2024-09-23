# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 21:13:19 2023

@author: user
"""

PARAMS = \
{
	"paths": {
		"data_path": '.\\files',
		"weight_path": 'generalized',
		"result_path": 'mint_nt',
		"plot_path_rawdata": 'mint_nt',
		"plot_path_results": 'mint_nt'
		},
	"save_weight": False,
	"save_result": True,
	"target_output": 'knee_ankle_l',
	"Trained_model": True,
	"manual_apgrf_flip": True,
	"add_shank_m_foot": True,
	"xcriteria": ['Acc_X', 'Acc_Y', 'Acc_Z', 'Gyr_X', 'Gyr_Y', 'Gyr_Z', 'rotx', 'roty', 'rotz' ],                   #'Mag_X', 'Mag_Y', 'Mag_Z', 'Mat11', 'Mat21', 'Mat31', 'Mat12','Mat22', 'Mat32', 'Mat13', 'Mat23', 'Mat33'
	"ycriteria": ['hip_flexion_r', 
			   'hip_adduction_r', 'hip_rotation_r', 'knee_angle_r', 'knee_angle_r_beta', 'ankle_angle_r', 'hip_flexion_l', 
			   'hip_adduction_l', 'hip_rotation_l', 'knee_angle_l', 'knee_angle_l_beta', 'ankle_angle_l'],
	"data_major_div": 5, # 1/number to test data
	"data_minor_add": 2, # add starting point of test data
	"plot_raw_data": False,
	"plot_result": False,

}


'''
Actual Position	Actual Velocity	Actual Current	Demanded Position	Demanded Velocity	Demanded Current	
bus_voltage	elmo_temp	motor_temp	F_PF	F_DF	Demanded_F_PF	Demanded_F_PF_NO_DF	estimated_pf_torque	estimated_df_torque	estimated_net_torque	
demanded_net_torque	flag_is_walking	walking_time	time_at_last_sync	value	gait_state	pull_state	handle_state	torque_mode	foot_angle_sagit_p	
foot_angle_front_p	foot_angle_trans_p	foot_lcl_acc_x_p	foot_lcl_acc_y_p	foot_lcl_acc_z_p	foot_gyro_sagit_p	foot_gyro_front_p	foot_gyro_trans_p	
foot_angle_sagit_np	foot_angle_front_np	foot_angle_trans_np	foot_lcl_acc_x_np	foot_lcl_acc_y_np	foot_lcl_acc_z_np	foot_gyro_sagit_np	foot_gyro_front_np	
foot_gyro_trans_np	shank_angle_sagit_p	shank_angle_front_p	shank_angle_trans_p	shank_lcl_acc_x_p	shank_lcl_acc_y_p	shank_lcl_acc_z_p	shank_gyro_sagit_p	
shank_gyro_front_p	shank_gyro_trans_p	shoulder_angle_sagit_np	shoulder_angle_front_np	shoulder_angle_trans_np	shoulder_lcl_acc_x_np	shoulder_lcl_acc_y_np	
shoulder_lcl_acc_z_np	shoulder_gyro_sagit_np	shoulder_gyro_front_np	shoulder_gyro_trans_np	pelvis_angle_sagit	pelvis_angle_front	pelvis_angle_trans	
pelvis_lcl_acc_x	pelvis_lcl_acc_y	pelvis_lcl_acc_z	pelvis_gyro_sagit	pelvis_gyro_front	pelvis_gyro_trans	stride_len_p	stride_len_np	walking_speed_p	
walking_speed_np	stride_time_p	stride_time_np	step_time_p	step_time_np	max_pf_p_support	max_df_p_support	max_exo_power_p_support	exo_power	
max_exo_net_torque_p_support	collection_mode	flag_initial_setting_completed
'''