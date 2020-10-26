# -*- coding: utf-8 -*-
"""Generate predictions using the 8 split datasets based on mass and jet number presence, we try ridge regression to predict outputs"""
import numpy as np
import matplotlib.pyplot as plt
from implementations import *
from proj1_helpers import *
from th_helpers import build_poly, split_data, compute_rmse
from th_ridge_regression import ridge_regression

# Best parameters found during our testing :
best_degrees = [9, 8, 9, 9, 8, 9, 9, 6]
best_lambdas = [0.00011508812515977024, 5.994842503189421e-09, 2.859091892859719e-06, 1e-10, 0.09088228539846002, 3.988415544392958e-05, 4.641588833612773e-08, 0.00180979262624709]
# For reference, these parameters should result in test correctness of :
# best_correctness=[81.04079143515382, 79.2655044298371, 83.1752055660974, 82.31029482841953, 83.6753077769993, 78.15321124580862, 79.7141722905915, 79.6028880866426]

# Load the train and test data from csv file
DATA_TRAIN_PATH = '../data/train.csv' 
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)
DATA_TEST_PATH = '../data/test.csv' 
_, tX_test, ids_test = load_csv_data(DATA_TEST_PATH)

# Variables used to remove features with error values based on physical 
# background of errors, explained in 
# https://higgsml.lal.in2p3.fr/files/2014/04/documentation_v1.8.pdf
y_jet = []
tx_jet = []
y_jet_nm = []
tx_jet_nm = []

# filtering according to undefinitions due to jet number
idx_jet_undef = [np.array([0,1,2,3,7,10,11,13,14,15,16,17,18,19,20,21,29]),
                np.array([0,1,2,3,7,8,9,10,11,13,14,15,16,17,18,19,20,21,23,24,25,29]),
                np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29]),
                np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29])]

# Extra filtering according to definition of mass
idx_jet_undef_nm = [np.array([1,2,3,7,10,11,13,14,15,16,17,18,19,20,21,29]),
                    np.array([1,2,3,7,8,9,10,11,13,14,15,16,17,18,19,20,21,23,24,25,29]),
                    np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29]),
                    np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,23,24,25,26,27,28,29])]


# ------------------------------------------------------------------------------------
# ------------------------------- Create 8 sub datasets absed on physics ------------- 
for jet in range(4):
    idx_jet = (tX[:,22]==jet) & (tX[:,0] != -999)
    y_jet.append(y[idx_jet])
    tx_jet.append(tX[idx_jet][:,idx_jet_undef[jet]])

# NB : no mass also has dupplicates from data that has mass, to have more data available for training.
for jet in range(4): 
    idx_jet = tX[:,22]==jet
    y_jet_nm.append(y[idx_jet])
    tx_jet_nm.append(tX[idx_jet][:,idx_jet_undef_nm[jet]])

# check whether the shapes are correct
# for jet in range(4):
#     print('Jet {:} shape is {:}'.format(jet,tx_jet[jet].shape))
#     print('Jet no mass {:} shape is {:}'.format(jet,tx_jet_nm[jet].shape))
# should output something like :
# Jet 0 shape is (73790, 17)
# Jet no mass 0 shape is (99913, 16)
# Jet 1 shape is (69982, 22)
# Jet no mass 1 shape is (77544, 21)
# Jet 2 shape is (47427, 29)
# Jet no mass 2 shape is (50379, 28)
# Jet 3 shape is (20687, 29)
# Jet no mass 3 shape is (22164, 28)
    
# Now put both 4 dim subsets into a single 8 dim array
idx_col_select_split = idx_jet_undef
y_split = []
tx_split_non_std = []

for jet in range(4):
    idx_jet = (tX[:,22]==jet) & (tX[:,0] != -999)
    y_split.append(y[idx_jet])
    tx_split_non_std.append(tX[idx_jet][:,idx_jet_undef[jet]])

for jet in range(4): # NB : no mass also has dupplicates from data that has mass, to have more data available for training.
    idx_jet = tX[:,22]==jet
    y_split.append(y[idx_jet])
    tx_split_non_std.append(tX[idx_jet][:,idx_jet_undef_nm[jet]])
    
for jet in range(4):
    idx_col_select_split.append(idx_jet_undef_nm[jet])
    
# Check results
# for set_i in range(8):
#     print(f'Set {set_i} shape is {tx_split_non_std[set_i].shape}')
# Should output something like :
# Set 0 shape is (73790, 17)
# Set 1 shape is (69982, 22)
# Set 2 shape is (47427, 29)
# Set 3 shape is (20687, 29)
# Set 4 shape is (99913, 16)
# Set 5 shape is (77544, 21)
# Set 6 shape is (50379, 28)
# Set 7 shape is (22164, 28)

# We tried standardizing the data, but in the end the best results
# achieved were without standardizing
tx_split = tx_split_non_std

# ------------------------------- Create 8 sub datasets absed on physics -------------
# ------------------------------------------------------------------------------------

# ------------------------------------------------------------------------------------
# -------------- Ridge regression to find wheights and create output file ------------

# We will first re-train our 8 models on the whole data, with the parameters found. As we checked for the best model complexity still avoiding overfitting, we will now train on all the train data. Because this is increasing the data on which the model trains, it should in fact reduce overfitting since we keep the same, therefore we do not split the data to check. We will in any case have AIcrowd to indicate if that was a good choice or not.

# First compute the model wheights with ridge regression on polynomial expanded sets
set_ws = []
for set_i in range(8):    
    train_x_aug = build_poly(tx_split[set_i], best_degrees[set_i])
    set_ws.append( ridge_regression(y_split[set_i], train_x_aug, best_lambdas[set_i]) )
    # Check the shapes for correct expansion
    # print(f"train_x_aug shape={train_x_aug.shape} w_shape={set_ws[set_i].shape}")
    # Should be something like :
    # train_x_aug shape=(73790, 154) w_shape=(154,)
    # train_x_aug shape=(69982, 177) w_shape=(177,)
    # train_x_aug shape=(47427, 262) w_shape=(262,)
    # train_x_aug shape=(20687, 262) w_shape=(262,)
    # train_x_aug shape=(99913, 129) w_shape=(129,)
    # train_x_aug shape=(77544, 190) w_shape=(190,)
    # train_x_aug shape=(50379, 253) w_shape=(253,)
    # train_x_aug shape=(22164, 169) w_shape=(169,)


# Then small function we will use to compute outputs.
# We do this by iterating through as we want to keep the right id for each datapoint,
# because the submission system needs the ids to identify each datapoint.
def compute_out_y(tX, ws, degrees):
    y_out = np.empty(tX.shape[0])

    for point_i, datapoint in enumerate(tX):
        jet_num = int(datapoint[22])
        if datapoint[0] != -999: # Mass is defined
            correct_set = datapoint[idx_jet_undef[jet_num]]
            correct_set_aug = build_poly(correct_set, degrees[jet_num])
            y_out[point_i] = np.dot(correct_set_aug, ws[jet_num])
        else : # Mass is undefined
            correct_set = datapoint[idx_jet_undef_nm[jet_num]]
            correct_set_aug = build_poly(correct_set, degrees[jet_num+4]) # offset of 4 because no mass ws start at index 4
            y_out[point_i] = np.dot(correct_set_aug, ws[jet_num+4])         

    y_out[y_out>=0] = 1
    y_out[y_out<0] = -1  
    
    return y_out

y_out_test = compute_out_y(tX_test, set_ws, best_degrees)
# Print the shapes to make sure they are the same, should output something like after the =>
# print(f"Y test out shape={y_out_test.shape} and original tX_test shape={tX_test.shape}")
# => Y test out shape=(568238,) and original tX_test shape=(568238, 30)

LAST_OUT_NAME = f"submit_ridge_split8sets_deg6-10_focusedLambdas.csv"
create_csv_submission(ids_test, y_out_test, LAST_OUT_NAME)
print("Correctly created file with name=", LAST_OUT_NAME)

# -------------- Ridge regression to find wheights and create output file ------------
# ------------------------------------------------------------------------------------