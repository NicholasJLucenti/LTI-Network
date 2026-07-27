import scipy.io
import numpy as np

data = scipy.io.loadmat(r'C:/Users/nickj/MATLAB Drive/LTI Network/NN_data/example_0001.mat')

print(data.keys())
print('resid_dx shape:', data['resid_dx'].shape)
print('resid_dy shape:', data['resid_dy'].shape)
print('Xi_ternary shape:', data['Xi_ternary'].shape)
print('topology:', data['topology'])
print('structure_label:', data['structure_label'])