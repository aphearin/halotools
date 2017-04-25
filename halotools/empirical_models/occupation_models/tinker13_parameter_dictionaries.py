"""
"""
import numpy as np


quiescent_fraction_control_masses = np.logspace(10.8, 14, 5)


param_dict_z2 = dict()

# active centrals
param_dict_z2['smhm_m1_active'] = 12.77
param_dict_z2['smhm_m0_active'] = 10.98
param_dict_z2['smhm_beta_active'] = 0.46
param_dict_z2['smhm_delta_active'] = 1.15
param_dict_z2['smhm_gamma_active'] = 2.15
param_dict_z2['scatter_model_param1_active'] = 0.24

# active satellites
param_dict_z2['bcut_active'] = 0.22
param_dict_z2['bsat_active'] = 24.55
param_dict_z2['betacut_active'] = 0.62
param_dict_z2['betasat_active'] = 1.16
param_dict_z2['alphasat_active'] = 0.96

# quiescent centrals
param_dict_z2['smhm_m1_quiescent'] = 12.18
param_dict_z2['smhm_m0_quiescent'] = 10.78
param_dict_z2['smhm_beta_quiescent'] = 0.13
param_dict_z2['smhm_delta_quiescent'] = 0.81
param_dict_z2['smhm_gamma_quiescent'] = 0.09
param_dict_z2['scatter_model_param1_quiescent'] = 0.21

# quiescent satellites
param_dict_z2['bcut_quiescent'] = 0.01
param_dict_z2['bsat_quiescent'] = 21.35
param_dict_z2['betacut_quiescent'] = -1.55
param_dict_z2['betasat_quiescent'] = 0.58
param_dict_z2['alphasat_quiescent'] = 1.15

# quiescent central fraction
param_dict_z2['quiescent_fraction_ordinates_param1'] = 10**-7.32
param_dict_z2['quiescent_fraction_ordinates_param2'] = 10**-1.17
param_dict_z2['quiescent_fraction_ordinates_param3'] = 0.47
param_dict_z2['quiescent_fraction_ordinates_param4'] = 0.68
param_dict_z2['quiescent_fraction_ordinates_param5'] = 0.81

param_dict_z1 = dict()

# active centrals
param_dict_z1['smhm_m1_active'] = 12.56
param_dict_z1['smhm_m0_active'] = 10.96
param_dict_z1['smhm_beta_active'] = 0.44
param_dict_z1['smhm_delta_active'] = 0.52
param_dict_z1['smhm_gamma_active'] = 1.48
param_dict_z1['scatter_model_param1_active'] = 0.21

# active satellites
param_dict_z1['bcut_active'] = 0.28
param_dict_z1['bsat_active'] = 33.96
param_dict_z1['betacut_active'] = 0.77
param_dict_z1['betasat_active'] = 1.05
param_dict_z1['alphasat_active'] = 0.99

# quiescent centrals
param_dict_z1['smhm_m1_quiescent'] = 12.08
param_dict_z1['smhm_m0_quiescent'] = 10.7
param_dict_z1['smhm_beta_quiescent'] = 0.32
param_dict_z1['smhm_delta_quiescent'] = 0.93
param_dict_z1['smhm_gamma_quiescent'] = 0.81
param_dict_z1['scatter_model_param1_quiescent'] = 0.28

# quiescent satellites
param_dict_z1['bcut_quiescent'] = 21.42
param_dict_z1['bsat_quiescent'] = 17.9
param_dict_z1['betacut_quiescent'] = -0.12
param_dict_z1['betasat_quiescent'] = 0.62
param_dict_z1['alphasat_quiescent'] = 1.08

# quiescent central fraction
param_dict_z1['quiescent_fraction_ordinates_param1'] = 10**-1.28
param_dict_z1['quiescent_fraction_ordinates_param2'] = 10**-0.85
param_dict_z1['quiescent_fraction_ordinates_param3'] = 0.54
param_dict_z1['quiescent_fraction_ordinates_param4'] = 0.63
param_dict_z1['quiescent_fraction_ordinates_param5'] = 0.77

param_dict_z3 = dict()

# active centrals
param_dict_z3['smhm_m1_active'] = 12.69
param_dict_z3['smhm_m0_active'] = 10.97
param_dict_z3['smhm_beta_active'] = 0.44
param_dict_z3['smhm_delta_active'] = 0.73
param_dict_z3['smhm_gamma_active'] = 4.71
param_dict_z3['scatter_model_param1_active'] = 0.25

# active satellites
param_dict_z3['bcut_active'] = 0.18
param_dict_z3['bsat_active'] = 112.7
param_dict_z3['betacut_active'] = 1.00
param_dict_z3['betasat_active'] = 2.65
param_dict_z3['alphasat_active'] = 0.84

# quiescent centrals
param_dict_z3['smhm_m1_quiescent'] = 12.21
param_dict_z3['smhm_m0_quiescent'] = 10.83
param_dict_z3['smhm_beta_quiescent'] = 0.02
param_dict_z3['smhm_delta_quiescent'] = 0.44
param_dict_z3['smhm_gamma_quiescent'] = 0.81
param_dict_z3['scatter_model_param1_quiescent'] = 0.18

# quiescent satellites
param_dict_z3['bcut_quiescent'] = 0.21
param_dict_z3['bsat_quiescent'] = 13.16
param_dict_z3['betacut_quiescent'] = 0.46
param_dict_z3['betasat_quiescent'] = 0.77
param_dict_z3['alphasat_quiescent'] = 0.98

# quiescent central fraction
param_dict_z3['quiescent_fraction_ordinates_param1'] = 10**-6.89
param_dict_z3['quiescent_fraction_ordinates_param2'] = 10**-1.23
param_dict_z3['quiescent_fraction_ordinates_param3'] = 0.43
param_dict_z3['quiescent_fraction_ordinates_param4'] = 0.59
param_dict_z3['quiescent_fraction_ordinates_param5'] = 0.76
