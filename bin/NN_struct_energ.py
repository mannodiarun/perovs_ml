import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from pandas import read_csv
import tensorflow.keras as keras
#import keras
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
import sklearn
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

import numpy as np     
import csv 
import copy 
import random 
#import mlpy
import pandas
import matplotlib.pyplot as plt 
#from mlpy import KernelRidge                                                                                                                                  
from sklearn.preprocessing import normalize




    # Read Data
ifile  = open('Data.csv', "rt")
reader = csv.reader(ifile)
csvdata=[]
for row in reader:
        csvdata.append(row)   
ifile.close()
numrow=len(csvdata)
numcol=len(csvdata[0]) 
csvdata = np.array(csvdata).reshape(numrow,numcol)
Index = csvdata[:,0]
Compounds = csvdata[:,1]
A = csvdata[:,2]
B = csvdata[:,3]
C = csvdata[:,4]
PBE_latt = csvdata[:,5]
PBE_form = csvdata[:,6]
HSE_latt = csvdata[:,7]
HSE_form = csvdata[:,8]
PBE_gap  = csvdata[:,9]
HSE_gap  = csvdata[:,10]
Ref_ind  = csvdata[:,11]
FOM      = csvdata[:,12]

Y = csvdata[:,5:9]
#Y = csvdata[:,9:13]
X = csvdata[:,13:]



    # Read Outside Data
ifile  = open('Outside_norm.csv', "rt")
reader = csv.reader(ifile)
csvdata=[]
for row in reader:
        csvdata.append(row)
ifile.close()
numrow=len(csvdata)
numcol=len(csvdata[0])
csvdata = np.array(csvdata).reshape(numrow,numcol)
Sys_out = csvdata[:,0]
A_out   = csvdata[:,1]
B_out   = csvdata[:,2]
C_out   = csvdata[:,3]
X_out = csvdata[:,4:]

n_out = C_out.size




    # Read Expt. Data Points
ifile  = open('Expt_points.csv', "rt")
reader = csv.reader(ifile)
csvdata=[]
for row in reader:
        csvdata.append(row)
ifile.close()
numrow=len(csvdata)
numcol=len(csvdata[0])
csvdata = np.array(csvdata).reshape(numrow,numcol)
Sys_expt = csvdata[:,0]
X_expt = csvdata[:,1:]

n_expt = Sys_expt.size




YY = copy.deepcopy(Y)
XX = copy.deepcopy(X)
n = C.size
m = int(X.size/n)
m_y = int(Y.size/n)
#m_y = 4

t = 0.20

X_train, X_test, Prop_train, Prop_test, Prop_train_pbe_latt, Prop_test_pbe_latt, Prop_train_pbe_form, Prop_test_pbe_form, Prop_train_hse_latt, Prop_test_hse_latt, Prop_train_hse_form, Prop_test_hse_form  =  train_test_split(XX, YY, PBE_latt, PBE_form, HSE_latt, HSE_form, test_size=t)

#X_train = copy.deepcopy(X)
#X_test  = copy.deepcopy(X)
#Prop_train = copy.deepcopy(prop)
#Prop_test  = copy.deepcopy(prop)

n_tr = int(Prop_train.size/m_y)
n_te = int(Prop_test.size/m_y)

X_train_fl = np.array(X_train, dtype="float32")
Prop_train_fl = np.array(Prop_train, dtype="float32")
X_test_fl = np.array(X_test, dtype="float32")
Prop_test_fl = np.array(Prop_test, dtype="float32")



PBE_latt_fl = [0.0]*n
PBE_form_fl = [0.0]*n
HSE_latt_fl = [0.0]*n
HSE_form_fl = [0.0]*n

for i in range(0,n):
    PBE_latt_fl[i] = float(PBE_latt[i])
    PBE_form_fl[i] = float(PBE_form[i])
    HSE_latt_fl[i] = float(HSE_latt[i])
    HSE_form_fl[i] = float(HSE_form[i])

max_range_pbe_latt = float ( np.max(PBE_latt_fl[:]) )
min_range_pbe_latt = float ( np.min(PBE_latt_fl[:]) )

max_range_pbe_form = float ( np.max(PBE_form_fl[:]) )
min_range_pbe_form = float ( np.min(PBE_form_fl[:]) )

max_range_hse_latt = float ( np.max(HSE_latt_fl[:]) )
min_range_hse_latt = float ( np.min(HSE_latt_fl[:]) )

max_range_hse_form = float ( np.max(HSE_form_fl[:]) )
min_range_hse_form = float ( np.min(HSE_form_fl[:]) )











## NN Optimizers and Model Definition


pipelines = []

parameters = [[0.0 for a in range(6)] for b in range(729)]

dp = [0.00, 0.10, 0.20]
n1 = [50, 75, 100]
n2 = [50, 75, 100]
lr = [0.001, 0.01, 0.1]
ep = [200, 400, 600]
bs = [50, 100, 200]

count = 0

for a in range(0,3):
    for b in range(0,3):
        for c in range(0,3):
            for d in range(0,3):
                for e in range(0,3):
                    for f in range(0,3):

                        parameters[count][0] = lr[a]
                        parameters[count][1] = n1[b]
                        parameters[count][2] = dp[c]
                        parameters[count][3] = n2[d]
                        parameters[count][4] = ep[e]
                        parameters[count][5] = bs[f]
                        count = count+1
                        
                        keras.optimizers.Adam(learning_rate=lr[a], beta_1=0.9, beta_2=0.999, amsgrad=False)

                        # define base model
                        def baseline_model():
                            model = Sequential()
                            model.add(Dense(m, input_dim=m, kernel_initializer='normal', activation='relu'))
                            model.add(Dense(n1[b], kernel_initializer='normal', activation='relu'))
                            model.add(Dropout(dp[c], input_shape=(m,)))
                            model.add(Dense(n2[d], kernel_initializer='normal', activation='relu'))
                            model.add(Dense(1, kernel_initializer='normal'))
                            model.compile(loss='mean_squared_error', optimizer='Adam')
                            return model

                        # evaluate model with standardized dataset
                        estimators = []
                        estimators.append(('standardize', StandardScaler()))
                        estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=ep[e], batch_size=bs[f], verbose=0)))
                        pipelines.append ( Pipeline(estimators) )




times = 1









## Train Model For PBE Lattice Constant ##


#times = 25
#times = len(pipelines)

train_errors = [0.0]*times
test_errors = [0.0]*times
nn_errors = list()

n_fold = 5
Prop_train_temp = np.array(Prop_train_pbe_latt, dtype="float32")
Prop_test_temp = np.array(Prop_test_pbe_latt, dtype="float32")

for i in range(0,times):
    pipeline = pipelines[np.random.randint(0,729)]
#    pipeline = pipelines[i]
    kf = KFold(n_splits = n_fold)
    mse_test_cv = 0.00
    mse_train_cv = 0.00
    for train, test in kf.split(X_train):
        X_train_cv, X_test_cv, Prop_train_cv, Prop_test_cv = X_train[train], X_train[test], Prop_train_temp[train], Prop_train_temp[test]

        X_train_cv_fl = np.array(X_train_cv, dtype="float32")
        Prop_train_cv_fl = np.array(Prop_train_cv, dtype="float32")
        X_test_cv_fl = np.array(X_test_cv, dtype="float32")
        Prop_test_cv_fl = np.array(Prop_test_cv, dtype="float32")

        pipeline.fit(X_train_cv_fl, Prop_train_cv_fl)
        Prop_pred_train_cv = pipeline.predict(X_train_cv_fl)
        Prop_pred_test_cv  = pipeline.predict(X_test_cv_fl)
        Pred_train_cv_fl = np.array(Prop_pred_train_cv, dtype="float32")
        Pred_test_cv_fl = np.array(Prop_pred_test_cv, dtype="float32")

        mse_test_cv = mse_test_cv  + sklearn.metrics.mean_squared_error(Prop_test_cv_fl, Pred_test_cv_fl)
        mse_train_cv = mse_train_cv + sklearn.metrics.mean_squared_error(Prop_train_cv_fl, Pred_train_cv_fl)
    mse_test = mse_test_cv / n_fold
    mse_train = mse_train_cv / n_fold
    train_errors[i] = mse_train
    test_errors[i] = mse_test
    nn_errors.append(pipeline)
i_opt = np.argmin(test_errors)
pipeline_opt = nn_errors[i_opt]

train_errors_pbe_latt = copy.deepcopy(train_errors)
test_errors_pbe_latt = copy.deepcopy(test_errors)

pipeline_opt.fit(X_train_fl, Prop_train_temp)
Pred_train = pipeline_opt.predict(X_train)
Pred_test  = pipeline_opt.predict(X_test)

Pred_train_pbe_latt_fl = [0.0]*n_tr
Pred_test_pbe_latt_fl  = [0.0]*n_te
Prop_train_pbe_latt_fl = [0.0]*n_tr
Prop_test_pbe_latt_fl  = [0.0]*n_te
for i in range(0,n_tr):
    Pred_train_pbe_latt_fl[i] = float(Pred_train[i])
    Prop_train_pbe_latt_fl[i] = float(Prop_train_pbe_latt[i])
for i in range(0,n_te):
    Pred_test_pbe_latt_fl[i] = float(Pred_test[i])
    Prop_test_pbe_latt_fl[i] = float(Prop_test_pbe_latt[i])


## Outside Predictions

Pred_out = pipeline_opt.predict(X_out)
Pred_out_pbe_latt = [0.0]*n_out
for i in range(0,n_out):
    Pred_out_pbe_latt[i] = float(Pred_out[i])

Pred_expt = pipeline_opt.predict(X_expt)
Pred_expt_pbe_latt = [0.0]*n_expt
for i in range(0,n_expt):
    Pred_expt_pbe_latt[i] = float(Pred_expt[i])










## Train Model For PBE Decomposition Energy ##


#times = 25
#times = len(pipelines)

train_errors = [0.0]*times
test_errors = [0.0]*times
nn_errors = list()

n_fold = 5
Prop_train_temp = np.array(Prop_train_pbe_form, dtype="float32")
Prop_test_temp = np.array(Prop_test_pbe_form, dtype="float32")

for i in range(0,times):
    pipeline = pipelines[np.random.randint(0,729)]
#    pipeline = pipelines[i]
    kf = KFold(n_splits = n_fold)
    mse_test_cv = 0.00
    mse_train_cv = 0.00
    for train, test in kf.split(X_train):
        X_train_cv, X_test_cv, Prop_train_cv, Prop_test_cv = X_train[train], X_train[test], Prop_train_temp[train], Prop_train_temp[test]

        X_train_cv_fl = np.array(X_train_cv, dtype="float32")
        Prop_train_cv_fl = np.array(Prop_train_cv, dtype="float32")
        X_test_cv_fl = np.array(X_test_cv, dtype="float32")
        Prop_test_cv_fl = np.array(Prop_test_cv, dtype="float32")

        pipeline.fit(X_train_cv_fl, Prop_train_cv_fl)
        Prop_pred_train_cv = pipeline.predict(X_train_cv_fl)
        Prop_pred_test_cv  = pipeline.predict(X_test_cv_fl)
        Pred_train_cv_fl = np.array(Prop_pred_train_cv, dtype="float32")
        Pred_test_cv_fl = np.array(Prop_pred_test_cv, dtype="float32")

        mse_test_cv = mse_test_cv  + sklearn.metrics.mean_squared_error(Prop_test_cv_fl, Pred_test_cv_fl)
        mse_train_cv = mse_train_cv + sklearn.metrics.mean_squared_error(Prop_train_cv_fl, Pred_train_cv_fl)
    mse_test = mse_test_cv / n_fold
    mse_train = mse_train_cv / n_fold
    train_errors[i] = mse_train
    test_errors[i] = mse_test
    nn_errors.append(pipeline)
i_opt = np.argmin(test_errors)
pipeline_opt = nn_errors[i_opt]

train_errors_pbe_form = copy.deepcopy(train_errors)
test_errors_pbe_form = copy.deepcopy(test_errors)

pipeline_opt.fit(X_train_fl, Prop_train_temp)
Pred_train = pipeline_opt.predict(X_train)
Pred_test  = pipeline_opt.predict(X_test)

Pred_train_pbe_form_fl = [0.0]*n_tr
Pred_test_pbe_form_fl  = [0.0]*n_te
Prop_train_pbe_form_fl = [0.0]*n_tr
Prop_test_pbe_form_fl  = [0.0]*n_te
for i in range(0,n_tr):
    Pred_train_pbe_form_fl[i] = float(Pred_train[i])
    Prop_train_pbe_form_fl[i] = float(Prop_train_pbe_form[i])
for i in range(0,n_te):
    Pred_test_pbe_form_fl[i] = float(Pred_test[i])
    Prop_test_pbe_form_fl[i] = float(Prop_test_pbe_form[i])


## Outside Predictions

Pred_out = pipeline_opt.predict(X_out)
Pred_out_pbe_form = [0.0]*n_out
for i in range(0,n_out):
    Pred_out_pbe_form[i] = float(Pred_out[i])

Pred_expt = pipeline_opt.predict(X_expt)
Pred_expt_pbe_form = [0.0]*n_expt
for i in range(0,n_expt):
    Pred_expt_pbe_form[i] = float(Pred_expt[i])









## Train Model For hse Lattice Constant ##


#times = 25
#times = len(pipelines)

train_errors = [0.0]*times
test_errors = [0.0]*times
nn_errors = list()

n_fold = 5
Prop_train_temp = np.array(Prop_train_hse_latt, dtype="float32")
Prop_test_temp = np.array(Prop_test_hse_latt, dtype="float32")

for i in range(0,times):
    pipeline = pipelines[np.random.randint(0,729)]
#    pipeline = pipelines[i]
    kf = KFold(n_splits = n_fold)
    mse_test_cv = 0.00
    mse_train_cv = 0.00
    for train, test in kf.split(X_train):
        X_train_cv, X_test_cv, Prop_train_cv, Prop_test_cv = X_train[train], X_train[test], Prop_train_temp[train], Prop_train_temp[test]

        X_train_cv_fl = np.array(X_train_cv, dtype="float32")
        Prop_train_cv_fl = np.array(Prop_train_cv, dtype="float32")
        X_test_cv_fl = np.array(X_test_cv, dtype="float32")
        Prop_test_cv_fl = np.array(Prop_test_cv, dtype="float32")

        pipeline.fit(X_train_cv_fl, Prop_train_cv_fl)
        Prop_pred_train_cv = pipeline.predict(X_train_cv_fl)
        Prop_pred_test_cv  = pipeline.predict(X_test_cv_fl)
        Pred_train_cv_fl = np.array(Prop_pred_train_cv, dtype="float32")
        Pred_test_cv_fl = np.array(Prop_pred_test_cv, dtype="float32")

        mse_test_cv = mse_test_cv  + sklearn.metrics.mean_squared_error(Prop_test_cv_fl, Pred_test_cv_fl)
        mse_train_cv = mse_train_cv + sklearn.metrics.mean_squared_error(Prop_train_cv_fl, Pred_train_cv_fl)
    mse_test = mse_test_cv / n_fold
    mse_train = mse_train_cv / n_fold
    train_errors[i] = mse_train
    test_errors[i] = mse_test
    nn_errors.append(pipeline)
i_opt = np.argmin(test_errors)
pipeline_opt = nn_errors[i_opt]

train_errors_hse_latt = copy.deepcopy(train_errors)
test_errors_hse_latt = copy.deepcopy(test_errors)

pipeline_opt.fit(X_train_fl, Prop_train_temp)
Pred_train = pipeline_opt.predict(X_train)
Pred_test  = pipeline_opt.predict(X_test)

Pred_train_hse_latt_fl = [0.0]*n_tr
Pred_test_hse_latt_fl  = [0.0]*n_te
Prop_train_hse_latt_fl = [0.0]*n_tr
Prop_test_hse_latt_fl  = [0.0]*n_te
for i in range(0,n_tr):
    Pred_train_hse_latt_fl[i] = float(Pred_train[i])
    Prop_train_hse_latt_fl[i] = float(Prop_train_hse_latt[i])
for i in range(0,n_te):
    Pred_test_hse_latt_fl[i] = float(Pred_test[i])
    Prop_test_hse_latt_fl[i] = float(Prop_test_hse_latt[i])


## Outside Predictions

Pred_out = pipeline_opt.predict(X_out)
Pred_out_hse_latt = [0.0]*n_out
for i in range(0,n_out):
    Pred_out_hse_latt[i] = float(Pred_out[i])

Pred_expt = pipeline_opt.predict(X_expt)
Pred_expt_hse_latt = [0.0]*n_expt
for i in range(0,n_expt):
    Pred_expt_hse_latt[i] = float(Pred_expt[i])










## Train Model For hse Decomposition Energy ##


#times = 25
#times = len(pipelines)

train_errors = [0.0]*times
test_errors = [0.0]*times
nn_errors = list()

n_fold = 5
Prop_train_temp = np.array(Prop_train_hse_form, dtype="float32")
Prop_test_temp = np.array(Prop_test_hse_form, dtype="float32")

for i in range(0,times):
    pipeline = pipelines[np.random.randint(0,729)]
#    pipeline = pipelines[i]
    kf = KFold(n_splits = n_fold)
    mse_test_cv = 0.00
    mse_train_cv = 0.00
    for train, test in kf.split(X_train):
        X_train_cv, X_test_cv, Prop_train_cv, Prop_test_cv = X_train[train], X_train[test], Prop_train_temp[train], Prop_train_temp[test]

        X_train_cv_fl = np.array(X_train_cv, dtype="float32")
        Prop_train_cv_fl = np.array(Prop_train_cv, dtype="float32")
        X_test_cv_fl = np.array(X_test_cv, dtype="float32")
        Prop_test_cv_fl = np.array(Prop_test_cv, dtype="float32")

        pipeline.fit(X_train_cv_fl, Prop_train_cv_fl)
        Prop_pred_train_cv = pipeline.predict(X_train_cv_fl)
        Prop_pred_test_cv  = pipeline.predict(X_test_cv_fl)
        Pred_train_cv_fl = np.array(Prop_pred_train_cv, dtype="float32")
        Pred_test_cv_fl = np.array(Prop_pred_test_cv, dtype="float32")

        mse_test_cv = mse_test_cv  + sklearn.metrics.mean_squared_error(Prop_test_cv_fl, Pred_test_cv_fl)
        mse_train_cv = mse_train_cv + sklearn.metrics.mean_squared_error(Prop_train_cv_fl, Pred_train_cv_fl)
    mse_test = mse_test_cv / n_fold
    mse_train = mse_train_cv / n_fold
    train_errors[i] = mse_train
    test_errors[i] = mse_test
    nn_errors.append(pipeline)
i_opt = np.argmin(test_errors)
pipeline_opt = nn_errors[i_opt]

train_errors_hse_form = copy.deepcopy(train_errors)
test_errors_hse_form = copy.deepcopy(test_errors)

pipeline_opt.fit(X_train_fl, Prop_train_temp)
Pred_train = pipeline_opt.predict(X_train)
Pred_test  = pipeline_opt.predict(X_test)

Pred_train_hse_form_fl = [0.0]*n_tr
Pred_test_hse_form_fl  = [0.0]*n_te
Prop_train_hse_form_fl = [0.0]*n_tr
Prop_test_hse_form_fl  = [0.0]*n_te
for i in range(0,n_tr):
    Pred_train_hse_form_fl[i] = float(Pred_train[i])
    Prop_train_hse_form_fl[i] = float(Prop_train_hse_form[i])
for i in range(0,n_te):
    Pred_test_hse_form_fl[i] = float(Pred_test[i])
    Prop_test_hse_form_fl[i] = float(Prop_test_hse_form[i])


## Outside Predictions

Pred_out = pipeline_opt.predict(X_out)
Pred_out_hse_form = [0.0]*n_out
for i in range(0,n_out):
    Pred_out_hse_form[i] = float(Pred_out[i])

Pred_expt = pipeline_opt.predict(X_expt)
Pred_expt_hse_form = [0.0]*n_expt
for i in range(0,n_expt):
    Pred_expt_hse_form[i] = float(Pred_expt[i])









errors = [[0.0 for a in range(8)] for b in range(times)]

for i in range(0,times):
    errors[i][0] = train_errors_pbe_latt[i]
    errors[i][1] = test_errors_pbe_latt[i]
    errors[i][2] = train_errors_pbe_form[i]
    errors[i][3] = test_errors_pbe_form[i]
    errors[i][4] = train_errors_hse_latt[i]
    errors[i][5] = test_errors_hse_latt[i]
    errors[i][6] = train_errors_hse_form[i]
    errors[i][7] = test_errors_hse_form[i]

#np.savetxt('errors.txt', errors)





Pred_out = [[0.0 for a in range(4)] for b in range(n_out)]

for i in range(0,n_out):
    Pred_out[i][0] = Pred_out_pbe_latt[i]
    Pred_out[i][1] = Pred_out_pbe_form[i]
    Pred_out[i][2] = Pred_out_hse_latt[i]
    Pred_out[i][3] = Pred_out_hse_form[i]

np.savetxt('Pred_out_struct_energ.txt', Pred_out)





Pred_expt = [[0.0 for a in range(4)] for b in range(n_expt)]

for i in range(0,n_expt):
    Pred_expt[i][0] = Pred_expt_pbe_latt[i]
    Pred_expt[i][1] = Pred_expt_pbe_form[i]
    Pred_expt[i][2] = Pred_expt_hse_latt[i]
    Pred_expt[i][3] = Pred_expt_hse_form[i]

np.savetxt('Pred_expt_struct_energ.txt', Pred_expt)









mse_test_prop  = sklearn.metrics.mean_squared_error(Prop_test_pbe_latt_fl, Pred_test_pbe_latt_fl)
mse_train_prop = sklearn.metrics.mean_squared_error(Prop_train_pbe_latt_fl, Pred_train_pbe_latt_fl)
rmse_test_pbe_latt  = np.sqrt(mse_test_prop)
rmse_train_pbe_latt = np.sqrt(mse_train_prop)
print('rmse_test_pbe_latt_const = ', np.sqrt(mse_test_prop))
print('rmse_train_pbe_latt_const = ', np.sqrt(mse_train_prop))
print('      ')

mse_test_prop  = sklearn.metrics.mean_squared_error(Prop_test_pbe_form_fl, Pred_test_pbe_form_fl)
mse_train_prop = sklearn.metrics.mean_squared_error(Prop_train_pbe_form_fl, Pred_train_pbe_form_fl)
rmse_test_pbe_form  = np.sqrt(mse_test_prop)
rmse_train_pbe_form = np.sqrt(mse_train_prop)
print('rmse_test_pbe_form_energy = ', np.sqrt(mse_test_prop))
print('rmse_train_pbe_form_energy = ', np.sqrt(mse_train_prop))
print('      ')

mse_test_prop  = sklearn.metrics.mean_squared_error(Prop_test_hse_latt_fl, Pred_test_hse_latt_fl)
mse_train_prop = sklearn.metrics.mean_squared_error(Prop_train_hse_latt_fl, Pred_train_hse_latt_fl)
rmse_test_hse_latt  = np.sqrt(mse_test_prop)
rmse_train_hse_latt = np.sqrt(mse_train_prop)
print('rmse_test_hse_latt_const = ', np.sqrt(mse_test_prop))
print('rmse_train_hse_latt_const = ', np.sqrt(mse_train_prop))
print('      ')

mse_test_prop  = sklearn.metrics.mean_squared_error(Prop_test_hse_form_fl, Pred_test_hse_form_fl)
mse_train_prop = sklearn.metrics.mean_squared_error(Prop_train_hse_form_fl, Pred_train_hse_form_fl)
rmse_test_hse_form  = np.sqrt(mse_test_prop)
rmse_train_hse_form = np.sqrt(mse_train_prop)
print('rmse_test_hse_form_energy = ', np.sqrt(mse_test_prop))
print('rmse_train_hse_form_energy = ', np.sqrt(mse_train_prop))
print('      ')

















## ML Parity Plots ##


#fig, ( [ax1, ax2], [ax3, ax4], [ax5, ax6] ) = plt.subplots( nrows=3, ncols=2, figsize=(6,6) )

#fig, ( [ax1, ax2], [ax3, ax4] ) = plt.subplots( nrows=2, ncols=2, sharex=True, sharey=True, figsize=(8,8) )

fig, ( [ax1, ax2], [ax3, ax4] ) = plt.subplots( nrows=2, ncols=2, figsize=(8,8) )

#fig, ( [ax1, ax2, ax3] ) = plt.subplots( nrows=1, ncols=3, figsize=(12,4) )

fig.text(0.5, 0.02, 'DFT Calculation', ha='center', fontsize=32)
fig.text(0.01, 0.5, 'ML Prediction', va='center', rotation='vertical', fontsize=32)

#fig, axes2d = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=(6,6))

plt.subplots_adjust(left=0.12, bottom=0.12, right=0.97, top=0.94, wspace=0.3, hspace=0.35)
plt.rc('font', family='Arial narrow')
#plt.tight_layout()
#plt.tight_layout(pad=0.6, w_pad=0.5, h_pad=0.5)

#plt.ylabel('ML Prediction', fontname='Arial Narrow', size=32)
#plt.xlabel('DFT Calculation', fontname='Arial Narrow', size=32)










Prop_train_temp = copy.deepcopy(Prop_train_pbe_latt_fl)
Pred_train_temp = copy.deepcopy(Pred_train_pbe_latt_fl)
Prop_test_temp  = copy.deepcopy(Prop_test_pbe_latt_fl)
Pred_test_temp  = copy.deepcopy(Pred_test_pbe_latt_fl)

a = [-175,0,125]
b = [-175,0,125]
ax1.plot(b, a, c='k', ls='-')

ax1.xaxis.set_tick_params(labelsize=20)
ax1.yaxis.set_tick_params(labelsize=20)

ax1.scatter(Prop_train_temp[:], Pred_train_temp[:], c='blue', marker='s', s=60, edgecolors='dimgrey', alpha=1.0, label='Training')
ax1.scatter(Prop_test_temp[:], Pred_test_temp[:], c='orange', marker='s', s=60, edgecolors='dimgrey', alpha=0.2, label='Test')

te = '%.2f' % rmse_test_pbe_latt
tr = '%.2f' % rmse_train_pbe_latt

ax1.text(6.15, 5.44, 'Test_rmse = ', c='r', fontsize=12)
ax1.text(6.69, 5.44, te, c='r', fontsize=12)
ax1.text(6.90, 5.43, '$\AA$', c='r', fontsize=12)
ax1.text(6.15, 5.28, 'Train_rmse = ', c='r', fontsize=12)
ax1.text(6.72, 5.28, tr, c='r', fontsize=12)
ax1.text(6.93, 5.27, '$\AA$', c='r', fontsize=12)

ax1.set_ylim([5.1, 7.1])
ax1.set_xlim([5.1, 7.1])
ax1.set_xticks([5.5, 6.0, 6.5, 7.0])
ax1.set_yticks([5.5, 6.0, 6.5, 7.0])

ax1.set_title('PBE Lattice Constant ($\AA$)', c='k', fontsize=20, pad=8)

ax1.legend(loc='upper left',ncol=1, frameon=True, prop={'family':'Arial narrow','size':12})









Prop_train_temp = copy.deepcopy(Prop_train_pbe_form_fl)
Pred_train_temp = copy.deepcopy(Pred_train_pbe_form_fl)
Prop_test_temp  = copy.deepcopy(Prop_test_pbe_form_fl)
Pred_test_temp  = copy.deepcopy(Pred_test_pbe_form_fl)


a = [-175,0,125]
b = [-175,0,125]
ax2.plot(b, a, c='k', ls='-')

ax2.xaxis.set_tick_params(labelsize=20)
ax2.yaxis.set_tick_params(labelsize=20)

ax2.scatter(Prop_train_temp[:], Pred_train_temp[:], c='blue', marker='s', s=60, edgecolors='dimgrey', alpha=1.0, label='Training')
ax2.scatter(Prop_test_temp[:], Pred_test_temp[:], c='orange', marker='s', s=60, edgecolors='dimgrey', alpha=0.2, label='Test')

te = '%.2f' % rmse_test_pbe_form
tr = '%.2f' % rmse_train_pbe_form

ax2.text(1.13, -0.71, 'Test_rmse = ', c='r', fontsize=12)
ax2.text(2.63, -0.71, te, c='r', fontsize=12)
ax2.text(3.18, -0.71, 'eV', c='r', fontsize=12)
ax2.text(1.03, -1.14, 'Train_rmse = ', c='r', fontsize=12)
ax2.text(2.63, -1.14, tr, c='r', fontsize=12)
ax2.text(3.18, -1.14, 'eV', c='r', fontsize=12)

ax2.set_ylim([-1.7, 3.8])
ax2.set_xlim([-1.7, 3.8])
ax2.set_xticks([-1.0, 0.0, 1.0, 2.0, 3.0])
ax2.set_yticks([-1.0, 0.0, 1.0, 2.0, 3.0])

ax2.set_title('PBE Decomposition Energy (eV)', c='k', fontsize=20, pad=8)

#ax2.legend(loc='upper left',ncol=1, frameon=True, prop={'family':'Arial narrow','size':12})








Prop_train_temp = copy.deepcopy(Prop_train_hse_latt_fl)
Pred_train_temp = copy.deepcopy(Pred_train_hse_latt_fl)
Prop_test_temp  = copy.deepcopy(Prop_test_hse_latt_fl)
Pred_test_temp  = copy.deepcopy(Pred_test_hse_latt_fl)

a = [-175,0,125]
b = [-175,0,125]
ax3.plot(b, a, c='k', ls='-')

ax3.xaxis.set_tick_params(labelsize=20)
ax3.yaxis.set_tick_params(labelsize=20)

ax3.scatter(Prop_train_temp[:], Pred_train_temp[:], c='blue', marker='s', s=60, edgecolors='dimgrey', alpha=1.0, label='Training')
ax3.scatter(Prop_test_temp[:], Pred_test_temp[:], c='orange', marker='s', s=60, edgecolors='dimgrey', alpha=0.2, label='Test')

te = '%.2f' % rmse_test_hse_latt
tr = '%.2f' % rmse_train_hse_latt

ax3.text(6.15, 5.44, 'Test_rmse = ', c='r', fontsize=12)
ax3.text(6.7, 5.44, te, c='r', fontsize=12)
ax3.text(6.9, 5.44, '$\AA$', c='r', fontsize=12)
ax3.text(6.15, 5.28, 'Train_rmse = ', c='r', fontsize=12)
ax3.text(6.73, 5.28, tr, c='r', fontsize=12)
ax3.text(6.93, 5.28, '$\AA$', c='r', fontsize=12)

ax3.set_ylim([5.1, 7.1])
ax3.set_xlim([5.1, 7.1])
ax3.set_xticks([5.5, 6.0, 6.5, 7.0])
ax3.set_yticks([5.5, 6.0, 6.5, 7.0])

ax3.set_title('HSE Lattice Constant ($\AA$)', c='k', fontsize=20, pad=8)

#ax3.legend(loc='upper left',ncol=1, frameon=True, prop={'family':'Arial narrow','size':12})








Prop_train_temp = copy.deepcopy(Prop_train_hse_form_fl)
Pred_train_temp = copy.deepcopy(Pred_train_hse_form_fl)
Prop_test_temp  = copy.deepcopy(Prop_test_hse_form_fl)
Pred_test_temp  = copy.deepcopy(Pred_test_hse_form_fl)


a = [-175,0,125]
b = [-175,0,125]
ax4.plot(b, a, c='k', ls='-')

ax4.xaxis.set_tick_params(labelsize=20)
ax4.yaxis.set_tick_params(labelsize=20)

ax4.scatter(Prop_train_temp[:], Pred_train_temp[:], c='blue', marker='s', s=60, edgecolors='dimgrey', alpha=1.0, label='Training')
ax4.scatter(Prop_test_temp[:], Pred_test_temp[:], c='orange', marker='s', s=60, edgecolors='dimgrey', alpha=0.2, label='Test')

te = '%.2f' % rmse_test_hse_form
tr = '%.2f' % rmse_train_hse_form

ax4.text(1.0, -0.7, 'Test_rmse = ', c='r', fontsize=12)
ax4.text(2.5, -0.7, te, c='r', fontsize=12)
ax4.text(3.05, -0.7, 'eV', c='r', fontsize=12)
ax4.text(0.9, -1.1, 'Train_rmse = ', c='r', fontsize=12)
ax4.text(2.5, -1.1, tr, c='r', fontsize=12)
ax4.text(3.05, -1.1, 'eV', c='r', fontsize=12)

ax4.set_ylim([-1.6, 3.7])
ax4.set_xlim([-1.6, 3.7])
ax4.set_xticks([-1.0, 0.0, 1.0, 2.0, 3.0])
ax4.set_yticks([-1.0, 0.0, 1.0, 2.0, 3.0])

ax4.set_title('HSE Decomposition Energy (eV)', c='k', fontsize=20, pad=8)

#ax4.legend(loc='upper left',ncol=1, frameon=True, prop={'family':'Arial narrow','size':12})












#plt.tick_params(axis='y', which='both', labelleft=True, labelright=False)

#plt.ylabel('ML Prediction', fontname='Arial Narrow', size=32)
#plt.xlabel('DFT Calculation', fontname='Arial Narrow', size=32)

#plt.rc('xtick', c='k', labelsize=16)
#plt.rc('ytick', c='k', labelsize=24)

plt.savefig('Parity_plot_struct_energ.eps', dpi=450)
plt.show()




