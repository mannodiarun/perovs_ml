# Regression Example With Boston Dataset: Standardized
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
import sklearn
import keras
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

#from __future__ import print_function
import numpy as np     
import csv 
import copy 
import random 
#import mlpy
import pandas
import matplotlib.pyplot as plt 
#from mlpy import KernelRidge                                                                                                                                                     



    # Read Data
ifile  = open('Defect_data.csv', "rt")
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

DFE_X_rich = csvdata[:,5]
EF_X_rich  = csvdata[:,6]
EF_frac_X_rich = csvdata[:,7]
DFE_mod = csvdata[:,8]
EF_mod  = csvdata[:,9]
EF_frac_mod = csvdata[:,10]
DFE_B_rich = csvdata[:,11]
EF_B_rich  = csvdata[:,12]
EF_frac_B_rich = csvdata[:,13]

V_A_form_X_rich = csvdata[:,14]
V_X_form_X_rich = csvdata[:,15]
V_A_form_mod = csvdata[:,16]
V_X_form_mod = csvdata[:,17]
V_A_form_B_rich = csvdata[:,18]
V_X_form_B_rich = csvdata[:,19]

CT_va = csvdata[:,20]
CT_vx = csvdata[:,21]
CT_frac_va = csvdata[:,22]
CT_frac_vx = csvdata[:,23]

X = csvdata[:,24:]

n = C.size
m = int(X.size/n)

Y = csvdata[:,8:12]
for i in range(0,n):
    Y[i][0] = copy.deepcopy(DFE_mod[i])
    Y[i][1] = copy.deepcopy(EF_mod[i])
    Y[i][2] = copy.deepcopy(CT_va[i])
    Y[i][3] = copy.deepcopy(CT_vx[i])
m_y = int(Y.size/n)

#Y = csvdata[:,8:12]
#for i in range(0,n):
#    Y[i][0] = copy.deepcopy(DFE_mod[i])
#    Y[i][1] = copy.deepcopy(EF_mod[i])
#    Y[i][2] = copy.deepcopy(CT_frac_va[i])
#    Y[i][3] = copy.deepcopy(CT_frac_vx[i])
#m_y = int(Y.size/n)



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



YY = copy.deepcopy(Y)
XX = copy.deepcopy(X)

t = 0.20

X_train, X_test, Prop_train, Prop_test, Prop_train_DFE, Prop_test_DFE, Prop_train_EF, Prop_test_EF, Prop_train_CT_va, Prop_test_CT_va, Prop_train_CT_vx, Prop_test_CT_vx  =  train_test_split(XX, YY, DFE_mod, EF_mod, CT_va, CT_vx, test_size=t)

#X_train, X_test, Prop_train, Prop_test, Prop_train_DFE, Prop_test_DFE, Prop_train_EF, Prop_test_EF, Prop_train_CT_va, Prop_test_CT_va, Prop_train_CT_vx, Prop_test_CT_vx  =  train_test_split(XX, YY, DFE_mod, EF_mod, CT_frac_va, CT_frac_vx, test_size=t)


n_tr = int(Prop_train.size/m_y)
n_te = int(Prop_test.size/m_y)


Prop_train_fl = [[0.0 for a in range(m_y)] for b in range(n_tr)]
for i in range(0,n_tr):
    for j in range(0,m_y):
        Prop_train_fl[i][j] = float(Prop_train[i][j])

Prop_test_fl = [[0.0 for a in range(m_y)] for b in range(n_te)]
for i in range(0,n_te):
    for j in range(0,m_y):
        Prop_test_fl[i][j] = float(Prop_test[i][j])
 
X_train_fl = [[0.0 for a in range(m)] for b in range(n_tr)]
for i in range(0,n_tr):
    for j in range(0,m):
        X_train_fl[i][j] = float(X_train[i][j])

X_test_fl = [[0.0 for a in range(m)] for b in range(n_te)]
for i in range(0,n_te):
    for j in range(0,m):
        X_test_fl[i][j] = float(X_test[i][j])




DFE_mod_fl = [0.0]*n
EF_mod_fl = [0.0]*n
CT_va_fl = [0.0]*n
CT_vx_fl = [0.0]*n

for i in range(0,n):
    DFE_mod_fl[i] = float(DFE_mod[i])
    EF_mod_fl[i] = float(EF_mod[i])
    CT_va_fl[i] = float(CT_frac_va[i])
    CT_vx_fl[i] = float(CT_frac_vx[i])

max_range_DFE = float ( np.max(DFE_mod_fl[:]) )
min_range_DFE = float ( np.min(DFE_mod_fl[:]) )

max_range_EF = float ( np.max(EF_mod_fl[:]) )
min_range_EF = float ( np.min(EF_mod_fl[:]) )

max_range_CT_va = float ( np.max(CT_va_fl[:]) )
min_range_CT_va = float ( np.min(CT_va_fl[:]) )

max_range_CT_vx = float ( np.max(CT_vx_fl[:]) )
min_range_CT_vx = float ( np.min(CT_vx_fl[:]) )










  ##  Train 4 Properties Together  ##  




## NN Optimizers and Model Definition


pipelines = []

parameters = [[0.0 for a in range(6)] for b in range(729)]

dp = [0.00, 0.10, 0.20]
n1 = [50, 100, 150]
n2 = [50, 100, 150]
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
                            model.add(Dense(m_y, kernel_initializer='normal'))
                            model.compile(loss='mean_squared_error', optimizer='Adam')
                            return model

                        # evaluate model with standardized dataset
                        estimators = []
                        estimators.append(('standardize', StandardScaler()))
                        estimators.append(('mlp', KerasRegressor(build_fn=baseline_model, epochs=ep[e], batch_size=bs[f], verbose=0)))
                        pipelines.append ( Pipeline(estimators) )














## Train Model On Data ##


times = 1
#times = len(pipelines)

train_errors = [0.0]*times
test_errors = [0.0]*times
nn_errors = list()

train_errors_DFE = [0.0]*times
test_errors_DFE = [0.0]*times
train_errors_EF = [0.0]*times
test_errors_EF = [0.0]*times
train_errors_CT_va = [0.0]*times
test_errors_CT_va = [0.0]*times
train_errors_CT_vx = [0.0]*times
test_errors_CT_vx = [0.0]*times

n_fold = 5

for i in range(0,times):
    pipeline = pipelines[np.random.randint(0,729)]
#    pipeline = pipelines[i]
    kf = KFold(n_splits = n_fold)
    mse_test_cv = 0.00
    mse_train_cv = 0.00
    mse_test_cv_DFE  = 0.00
    mse_train_cv_DFE = 0.00
    mse_test_cv_EF  = 0.00
    mse_train_cv_EF = 0.00
    mse_test_cv_CT_va   = 0.00
    mse_train_cv_CT_va  = 0.00
    mse_test_cv_CT_vx   = 0.00
    mse_train_cv_CT_vx  = 0.00

    for train, test in kf.split(X_train):
        X_train_cv, X_test_cv, Prop_train_cv, Prop_test_cv = X_train[train], X_train[test], Prop_train[train], Prop_train[test]
        X_train_cv_fl = np.array(X_train_cv, dtype="float32")
        X_test_cv_fl = np.array(X_test_cv, dtype="float32")
        Prop_train_cv_fl = np.array(Prop_train_cv, dtype="float32")
        Prop_test_cv_fl = np.array(Prop_test_cv, dtype="float32")
        pipeline.fit(X_train_cv_fl, Prop_train_cv_fl)
        Prop_pred_train_cv = pipeline.predict(X_train_cv_fl)
        Prop_pred_test_cv  = pipeline.predict(X_test_cv_fl)

        n_te_cv = int(Prop_test_cv.size/m_y)
        n_tr_cv = int(Prop_train_cv.size/m_y)

        Prop_test_cv_DFE_fl = [0.0]*n_te_cv
        Prop_test_cv_EF_fl = [0.0]*n_te_cv
        Prop_test_cv_CT_va_fl = [0.0]*n_te_cv
        Prop_test_cv_CT_vx_fl = [0.0]*n_te_cv
        for aa in range(0,n_te_cv):
            Prop_test_cv_DFE_fl[aa] = float(Prop_test_cv[aa][0])
            Prop_test_cv_EF_fl[aa] = float(Prop_test_cv[aa][1])
            Prop_test_cv_CT_va_fl[aa] = float(Prop_test_cv[aa][2])
            Prop_test_cv_CT_vx_fl[aa] = float(Prop_test_cv[aa][3])

        Pred_test_cv_DFE_fl = [0.0]*n_te_cv
        Pred_test_cv_EF_fl = [0.0]*n_te_cv
        Pred_test_cv_CT_va_fl = [0.0]*n_te_cv
        Pred_test_cv_CT_vx_fl = [0.0]*n_te_cv
        for aa in range(0,n_te_cv):
            Pred_test_cv_DFE_fl[aa] = float(Prop_pred_test_cv[aa][0])
            Pred_test_cv_EF_fl[aa] = float(Prop_pred_test_cv[aa][1])
            Pred_test_cv_CT_va_fl[aa] = float(Prop_pred_test_cv[aa][2])
            Pred_test_cv_CT_vx_fl[aa] = float(Prop_pred_test_cv[aa][3])

        Prop_train_cv_DFE_fl = [0.0]*n_tr_cv
        Prop_train_cv_EF_fl = [0.0]*n_tr_cv
        Prop_train_cv_CT_va_fl = [0.0]*n_tr_cv
        Prop_train_cv_CT_vx_fl = [0.0]*n_tr_cv
        for aa in range(0,n_tr_cv):
            Prop_train_cv_DFE_fl[aa] = float(Prop_train_cv[aa][0])
            Prop_train_cv_EF_fl[aa] = float(Prop_train_cv[aa][1])
            Prop_train_cv_CT_va_fl[aa] = float(Prop_train_cv[aa][2])
            Prop_train_cv_CT_vx_fl[aa] = float(Prop_train_cv[aa][3])

        Pred_train_cv_DFE_fl = [0.0]*n_tr_cv
        Pred_train_cv_EF_fl = [0.0]*n_tr_cv
        Pred_train_cv_CT_va_fl = [0.0]*n_tr_cv
        Pred_train_cv_CT_vx_fl = [0.0]*n_tr_cv
        for aa in range(0,n_tr_cv):
            Pred_train_cv_DFE_fl[aa] = float(Prop_pred_train_cv[aa][0])
            Pred_train_cv_EF_fl[aa] = float(Prop_pred_train_cv[aa][1])
            Pred_train_cv_CT_va_fl[aa] = float(Prop_pred_train_cv[aa][2])
            Pred_train_cv_CT_vx_fl[aa] = float(Prop_pred_train_cv[aa][3])

        mse_test_cv_DFE  = mse_test_cv_DFE + sklearn.metrics.mean_squared_error(Prop_test_cv_DFE_fl, Pred_test_cv_DFE_fl)
        mse_train_cv_DFE = mse_train_cv_DFE + sklearn.metrics.mean_squared_error(Prop_train_cv_DFE_fl, Pred_train_cv_DFE_fl)

        mse_test_cv_EF  = mse_test_cv_EF + sklearn.metrics.mean_squared_error(Prop_test_cv_EF_fl, Pred_test_cv_EF_fl)
        mse_train_cv_EF = mse_train_cv_EF + sklearn.metrics.mean_squared_error(Prop_train_cv_EF_fl, Pred_train_cv_EF_fl)

        mse_test_cv_CT_va  = mse_test_cv_CT_va + sklearn.metrics.mean_squared_error(Prop_test_cv_CT_va_fl, Pred_test_cv_CT_va_fl)
        mse_train_cv_CT_va = mse_train_cv_CT_va + sklearn.metrics.mean_squared_error(Prop_train_cv_CT_va_fl, Pred_train_cv_CT_va_fl)

        mse_test_cv_CT_vx  = mse_test_cv_CT_vx + sklearn.metrics.mean_squared_error(Prop_test_cv_CT_vx_fl, Pred_test_cv_CT_vx_fl)
        mse_train_cv_CT_vx = mse_train_cv_CT_vx + sklearn.metrics.mean_squared_error(Prop_train_cv_CT_vx_fl, Pred_train_cv_CT_vx_fl)

        mse_test_cv = mse_test_cv + np.sqrt ( np.power(mse_test_cv_DFE/(max_range_DFE - min_range_DFE),2)  +  np.power(mse_test_cv_EF/(max_range_EF - min_range_EF),2)  +  np.power(mse_test_cv_CT_va/(max_range_CT_va - min_range_CT_va),2) + np.power(mse_test_cv_CT_vx/(max_range_CT_vx - min_range_CT_vx),2) )
        mse_train_cv = mse_train_cv + np.sqrt ( np.power(mse_train_cv_DFE/(max_range_DFE - min_range_DFE),2)  +  np.power(mse_train_cv_EF/(max_range_EF - min_range_EF),2)  +  np.power(mse_train_cv_CT_va/(max_range_CT_va - min_range_CT_va),2) + np.power(mse_train_cv_CT_vx/(max_range_CT_vx - min_range_CT_vx),2) )

    mse_test = mse_test_cv / n_fold
    mse_train = mse_train_cv / n_fold
    train_errors[i] = mse_train
    test_errors[i] = mse_test
    train_errors_DFE[i] = np.sqrt ( mse_train_cv_DFE / n_fold )
    test_errors_DFE[i]  = np.sqrt ( mse_test_cv_DFE / n_fold )
    train_errors_EF[i] = np.sqrt ( mse_train_cv_EF / n_fold )
    test_errors_EF[i]  = np.sqrt ( mse_test_cv_EF / n_fold )
    train_errors_CT_va[i]  = np.sqrt ( mse_train_cv_CT_va / n_fold )
    test_errors_CT_va[i]   = np.sqrt ( mse_test_cv_CT_va / n_fold )
    train_errors_CT_vx[i]  = np.sqrt ( mse_train_cv_CT_vx / n_fold )
    test_errors_CT_vx[i]   = np.sqrt ( mse_test_cv_CT_vx / n_fold )
    nn_errors.append(pipeline)

i_opt = np.argmin(test_errors)
pipeline_opt = nn_errors[i]

X_train_fl = np.array(X_train, dtype="float32")
X_test_fl = np.array(X_test, dtype="float32")
Prop_train_fl = np.array(Prop_train, dtype="float32")
Prop_test_fl = np.array(Prop_test, dtype="float32")
pipeline_opt.fit(X_train_fl, Prop_train_fl)
Pred_train = pipeline_opt.predict(X_train_fl)
Pred_test  = pipeline_opt.predict(X_test_fl)

Pred_train_fl = [[0.0 for a in range(m_y)] for b in range(n_tr)]
Pred_test_fl = [[0.0 for a in range(m_y)] for b in range(n_te)]
for i in range(0,n_tr):
    for j in range(0,m_y):
        Pred_train_fl[i][j] = float(Pred_train[i][j])
for i in range(0,n_te):
    for j in range(0,m_y):
        Pred_test_fl[i][j] = float(Pred_test[i][j])


errors_all = [[0.0 for a in range(8)] for b in range(times)]
for i in range(0,times):
    errors_all[i][0] = train_errors_DFE[i]
    errors_all[i][1] = test_errors_DFE[i]
    errors_all[i][2] = train_errors_EF[i]
    errors_all[i][3] = test_errors_EF[i]
    errors_all[i][4] = train_errors_CT_va[i]
    errors_all[i][5] = test_errors_CT_va[i]
    errors_all[i][6] = train_errors_CT_vx[i]
    errors_all[i][7] = test_errors_CT_vx[i]

#np.savetxt('errors.txt', errors_all)
#np.savetxt('parameters.txt', parameters)










Prop_train_DFE = [0.0]*n_tr
Pred_train_DFE = [0.0]*n_tr
Prop_test_DFE  = [0.0]*n_te
Pred_test_DFE  = [0.0]*n_te

for i in range(0,n_tr):
    Prop_train_DFE[i] = Prop_train_fl[i][0] 
    Pred_train_DFE[i] = Pred_train_fl[i][0]

for i in range(0,n_te):
    Prop_test_DFE[i] = Prop_test_fl[i][0]          
    Pred_test_DFE[i] = Pred_test_fl[i][0]



Prop_train_EF = [0.0]*n_tr
Pred_train_EF = [0.0]*n_tr
Prop_test_EF  = [0.0]*n_te
Pred_test_EF  = [0.0]*n_te

for i in range(0,n_tr):
    Prop_train_EF[i] = Prop_train_fl[i][1]
    Pred_train_EF[i] = Pred_train_fl[i][1]

for i in range(0,n_te):
    Prop_test_EF[i] = Prop_test_fl[i][1] 
    Pred_test_EF[i] = Pred_test_fl[i][1]



Prop_train_CT_va = [0.0]*n_tr
Pred_train_CT_va = [0.0]*n_tr
Prop_test_CT_va  = [0.0]*n_te
Pred_test_CT_va  = [0.0]*n_te

for i in range(0,n_tr):
    Prop_train_CT_va[i] = Prop_train_fl[i][2] 
    Pred_train_CT_va[i] = Pred_train_fl[i][2]

for i in range(0,n_te):
    Prop_test_CT_va[i] = Prop_test_fl[i][2]          
    Pred_test_CT_va[i] = Pred_test_fl[i][2]



Prop_train_CT_vx = [0.0]*n_tr
Pred_train_CT_vx = [0.0]*n_tr
Prop_test_CT_vx  = [0.0]*n_te
Pred_test_CT_vx  = [0.0]*n_te

for i in range(0,n_tr):
    Prop_train_CT_vx[i] = Prop_train_fl[i][3]
    Pred_train_CT_vx[i] = Pred_train_fl[i][3]

for i in range(0,n_te):
    Prop_test_CT_vx[i] = Prop_test_fl[i][3]          
    Pred_test_CT_vx[i] = Pred_test_fl[i][3]






mse_test_prop  = sklearn.metrics.mean_squared_error(Prop_test_DFE, Pred_test_DFE)
mse_train_prop = sklearn.metrics.mean_squared_error(Prop_train_DFE, Pred_train_DFE)
rmse_test_DFE  = np.sqrt(mse_test_prop)
rmse_train_DFE = np.sqrt(mse_train_prop)
print('rmse_test_DFE = ', np.sqrt(mse_test_prop))
print('rmse_train_DFE = ', np.sqrt(mse_train_prop))
print('      ')

mse_test_prop  = sklearn.metrics.mean_squared_error(Prop_test_EF, Pred_test_EF)
mse_train_prop = sklearn.metrics.mean_squared_error(Prop_train_EF, Pred_train_EF)
rmse_test_EF  = np.sqrt(mse_test_prop)
rmse_train_EF = np.sqrt(mse_train_prop)
print('rmse_test_EF = ', np.sqrt(mse_test_prop))
print('rmse_train_EF = ', np.sqrt(mse_train_prop))
print('      ')

mse_test_prop  = sklearn.metrics.mean_squared_error(Prop_test_CT_va, Pred_test_CT_va)
mse_train_prop = sklearn.metrics.mean_squared_error(Prop_train_CT_va, Pred_train_CT_va)
rmse_test_CT_va  = np.sqrt(mse_test_prop)
rmse_train_CT_va = np.sqrt(mse_train_prop)
print('rmse_test_CT_va = ', np.sqrt(mse_test_prop))
print('rmse_train_CT_va = ', np.sqrt(mse_train_prop))
print('      ')

mse_test_prop  = sklearn.metrics.mean_squared_error(Prop_test_CT_vx, Pred_test_CT_vx)
mse_train_prop = sklearn.metrics.mean_squared_error(Prop_train_CT_vx, Pred_train_CT_vx)
rmse_test_CT_vx  = np.sqrt(mse_test_prop)
rmse_train_CT_vx = np.sqrt(mse_train_prop)
print('rmse_test_CT_vx = ', np.sqrt(mse_test_prop))
print('rmse_train_CT_vx = ', np.sqrt(mse_train_prop))
print('      ')

















## Outside Predictions




Pred_out = pipeline_opt.predict(X_out)
Pred_out_fl = [[0.0 for a in range(m_y)] for b in range(n_out)]
for i in range(0,n_out):
    for j in range(0,m_y):
        Pred_out_fl[i][j] = float(Pred_out[i][j])

np.savetxt('Pred_out_defect.csv',Pred_out)















## ML Parity Plots ##


#fig, ( [ax1, ax2], [ax3, ax4], [ax5, ax6] ) = plt.subplots( nrows=3, ncols=2, figsize=(6,6) )

#fig, ( [ax1, ax2], [ax3, ax4] ) = plt.subplots( nrows=2, ncols=2, sharex=True, sharey=True, figsize=(8,8) )

fig, ( [ax1, ax2], [ax3, ax4] ) = plt.subplots( nrows=2, ncols=2, figsize=(8,8) )

fig.text(0.5, 0.02, 'DFT Calculation', ha='center', fontsize=32)
fig.text(0.02, 0.5, 'ML Prediction', va='center', rotation='vertical', fontsize=32)

#fig, axes2d = plt.subplots(nrows=3, ncols=3, sharex=True, sharey=True, figsize=(6,6))

plt.subplots_adjust(left=0.14, bottom=0.12, right=0.96, top=0.94, wspace=0.3, hspace=0.3)
plt.rc('font', family='Arial narrow')
#plt.tight_layout()
#plt.tight_layout(pad=0.6, w_pad=0.5, h_pad=0.5)

#plt.ylabel('ML Prediction', fontname='Arial Narrow', size=32)
#plt.xlabel('DFT Calculation', fontname='Arial Narrow', size=32)







Prop_train_temp = copy.deepcopy(Prop_train_DFE)
Pred_train_temp = copy.deepcopy(Pred_train_DFE)
Prop_test_temp  = copy.deepcopy(Prop_test_DFE)
Pred_test_temp  = copy.deepcopy(Pred_test_DFE)

a = [-175,0,125]
b = [-175,0,125]
ax1.plot(b, a, c='k', ls='-')

#ax1.set_ylabel('ML Prediction', fontname='Arial Narrow', size=20)
#ax1.set_xlabel('DFT Calculation', fontname='Arial Narrow', size=20)
#ax1.rc('xtick', labelsize=12)
#ax1.rc('ytick', labelsize=12)

ax1.xaxis.set_tick_params(labelsize=20)
ax1.yaxis.set_tick_params(labelsize=20)

ax1.scatter(Prop_train_temp[:], Pred_train_temp[:], c='blue', marker='s', s=60, edgecolors='dimgrey', alpha=1.0, label='Training')
ax1.scatter(Prop_test_temp[:], Pred_test_temp[:], c='orange', marker='s', s=60, edgecolors='dimgrey', alpha=0.2, label='Test')

te = '%.2f' % rmse_test_DFE
tr = '%.2f' % rmse_train_DFE

ax1.text(0.70, -0.28, 'Test_rmse = ', c='r', fontsize=12)
ax1.text(1.60, -0.28, te, c='r', fontsize=12)
ax1.text(1.95, -0.28, 'eV', c='r', fontsize=12)
ax1.text(0.65, -0.52, 'Train_rmse = ', c='r', fontsize=12)
ax1.text(1.60, -0.52, tr, c='r', fontsize=12)
ax1.text(1.95, -0.52, 'eV', c='r', fontsize=12)

ax1.set_ylim([-0.8, 2.3])
ax1.set_xlim([-0.8, 2.3])
ax1.set_xticks([-0.5, 0.25, 1, 1.75])
ax1.set_yticks([-0.5, 0.25, 1, 1.75])

ax1.set_title('Eqm. Defect Formation Energy (eV)', c='k', fontsize=18, pad=7)

ax1.legend(loc='upper left',ncol=1, frameon=True, prop={'family':'Arial narrow','size':12})
#plt.savefig('plot_CT1.eps', dpi=450)
#plt.show()









Prop_train_temp = copy.deepcopy(Prop_train_EF)
Pred_train_temp = copy.deepcopy(Pred_train_EF)
Prop_test_temp  = copy.deepcopy(Prop_test_EF)
Pred_test_temp  = copy.deepcopy(Pred_test_EF)

a = [-175,0,125]
b = [-175,0,125]
ax2.plot(b, a, c='k', ls='-')

#ax2.set_ylabel('ML Prediction', fontname='Arial Narrow', size=20)
#ax2.set_xlabel('DFT Calculation', fontname='Arial Narrow', size=20)
#ax2.rc('xtick', labelsize=12)
#ax2.rc('ytick', labelsize=12)

ax2.xaxis.set_tick_params(labelsize=20)
ax2.yaxis.set_tick_params(labelsize=20)

ax2.scatter(Prop_train_temp[:], Pred_train_temp[:], c='blue', marker='s', s=60, edgecolors='dimgrey', alpha=1.0, label='Training')
ax2.scatter(Prop_test_temp[:], Pred_test_temp[:], c='orange', marker='s', s=60, edgecolors='dimgrey', alpha=0.2, label='Test')

te = '%.2f' % rmse_test_EF
tr = '%.2f' % rmse_train_EF

ax2.text(0.6, -1.05, 'Test_rmse = ', c='r', fontsize=12)
ax2.text(2.05, -1.05, te, c='r', fontsize=12)
ax2.text(2.58, -1.05, 'eV', c='r', fontsize=12)
ax2.text(0.55, -1.5, 'Train_rmse = ', c='r', fontsize=12)
ax2.text(2.1, -1.5, tr, c='r', fontsize=12)
ax2.text(2.62, -1.5, 'eV', c='r', fontsize=12)

ax2.set_ylim([-2.0, 3.2])
ax2.set_xlim([-2.0, 3.2])
#ax2.set_ylim([-1.8, 1.0])
#ax2.set_xlim([-1.8, 1.0])
ax2.set_xticks([-1.5, 0, 1.5, 3.0])
ax2.set_yticks([-1.5, 0, 1.5, 3.0])

ax2.set_title('Eqm. Fermi Level (eV)', c='k', fontsize=18, pad=7)

#ax2.legend(loc='upper left',ncol=1, frameon=True, prop={'family':'Arial narrow','size':12})
#plt.savefig('plot_CT2.eps', dpi=450)
#plt.show()










Prop_train_temp = copy.deepcopy(Prop_train_CT_va)
Pred_train_temp = copy.deepcopy(Pred_train_CT_va)
Prop_test_temp  = copy.deepcopy(Prop_test_CT_va)
Pred_test_temp  = copy.deepcopy(Pred_test_CT_va)

a = [-175,0,125]
b = [-175,0,125]
ax3.plot(b, a, c='k', ls='-')

#ax3.set_ylabel('ML Prediction', fontname='Arial Narrow', size=20)
#ax3.set_xlabel('DFT Calculation', fontname='Arial Narrow', size=20)
#ax3.rc('xtick', labelsize=12)
#ax3.rc('ytick', labelsize=12)

ax3.xaxis.set_tick_params(labelsize=20)
ax3.yaxis.set_tick_params(labelsize=20)

ax3.scatter(Prop_train_temp[:], Pred_train_temp[:], c='blue', marker='s', s=60, edgecolors='dimgrey', alpha=1.0, label='Training')
ax3.scatter(Prop_test_temp[:], Pred_test_temp[:], c='orange', marker='s', s=60, edgecolors='dimgrey', alpha=0.2, label='Test')

te = '%.2f' % rmse_test_CT_va
tr = '%.2f' % rmse_train_CT_va


## Actual Level ##

ax3.text(0.05, -0.62, 'Test_rmse = ', c='r', fontsize=12)
ax3.text(0.60, -0.62, te, c='r', fontsize=12)
ax3.text(0.81, -0.62, 'eV', c='r', fontsize=12)
ax3.text(0.02, -0.80, 'Train_rmse = ', c='r', fontsize=12)
ax3.text(0.62, -0.80, tr, c='r', fontsize=12)
ax3.text(0.83, -0.80, 'eV', c='r', fontsize=12)

ax3.set_ylim([-1.0, 1.0])
ax3.set_xlim([-1.0, 1.0])
ax3.set_xticks([-0.75, -0.25, 0.25, 0.75])
ax3.set_yticks([-0.75, -0.25, 0.25, 0.75])


## Fractional Level ##

#ax3.text(-0.42, -1.3, 'Test_rmse = ', c='r', fontsize=12)
#ax3.text(0.25, -1.3, te, c='r', fontsize=12)
#ax3.text(0.5, -1.3, 'eV', c='r', fontsize=12)
#ax3.text(-0.45, -1.5, 'Train_rmse = ', c='r', fontsize=12)
#ax3.text(0.25, -1.5, tr, c='r', fontsize=12)
#ax3.text(0.5, -1.5, 'eV', c='r', fontsize=12)

#ax3.set_ylim([-1.7, 0.7])
#ax3.set_xlim([-1.7, 0.7])
#ax3.set_xticks([-1.5, -1.0, -0.5, 0.0, 0.5])
#ax3.set_yticks([-1.5, -1.0, -0.5, 0.0, 0.5])


ax3.set_title('V$_{A}$ (0/-1) Transition Level (eV)', c='k', fontsize=18, pad=7)

#ax3.legend(loc='upper left',ncol=1, frameon=True, prop={'family':'Arial narrow','size':12})
#ax3.savefig('plot_CT3.eps', dpi=450)
#ax3.show()










Prop_train_temp = copy.deepcopy(Prop_train_CT_vx)
Pred_train_temp = copy.deepcopy(Pred_train_CT_vx)
Prop_test_temp  = copy.deepcopy(Prop_test_CT_vx)
Pred_test_temp  = copy.deepcopy(Pred_test_CT_vx)


a = [-175,0,125]
b = [-175,0,125]
ax4.plot(b, a, c='k', ls='-')

#ax4.set_ylabel('ML Prediction', fontname='Arial Narrow', size=20)
#ax4.set_xlabel('DFT Calculation', fontname='Arial Narrow', size=20)
#ax4.rc('xtick', labelsize=12)
#ax4.rc('ytick', labelsize=12)

ax4.xaxis.set_tick_params(labelsize=20)
ax4.yaxis.set_tick_params(labelsize=20)

ax4.scatter(Prop_train_temp[:], Pred_train_temp[:], c='blue', marker='s', s=60, edgecolors='dimgrey', alpha=1.0, label='Training')
ax4.scatter(Prop_test_temp[:], Pred_test_temp[:], c='orange', marker='s', s=60, edgecolors='dimgrey', alpha=0.2, label='Test')

te = '%.2f' % rmse_test_CT_vx
tr = '%.2f' % rmse_train_CT_vx


## Actual Levels ##

ax4.text(2.32, 0.6, 'Test_rmse = ', c='r', fontsize=12)
ax4.text(3.70, 0.6, te, c='r', fontsize=12)
ax4.text(4.22, 0.6, 'eV', c='r', fontsize=12)
ax4.text(2.30, 0.2, 'Train_rmse = ', c='r', fontsize=12)
ax4.text(3.75, 0.2, tr, c='r', fontsize=12)
ax4.text(4.25, 0.2, 'eV', c='r', fontsize=12)

ax4.set_ylim([-0.3, 4.7])
ax4.set_xlim([-0.3, 4.7])
ax4.set_xticks([0, 1, 2, 3, 4])
ax4.set_yticks([0, 1, 2, 3, 4])


## Fractional Levels ##

#ax4.text(0.49, -0.1, 'Test_rmse = ', c='r', fontsize=12)
#ax4.text(1.03, -0.1, te, c='r', fontsize=12)
#ax4.text(1.22, -0.1, 'eV', c='r', fontsize=12)
#ax4.text(0.46, -0.25, 'Train_rmse = ', c='r', fontsize=12)
#ax4.text(0.99, -0.25, tr, c='r', fontsize=12)
#ax4.text(1.17, -0.25, 'eV', c='r', fontsize=12)

#ax4.set_ylim([-0.4, 1.4])
#ax4.set_xlim([-0.4, 1.4])
#ax4.set_xticks([0, 0.5, 1.0])
#ax4.set_yticks([0, 0.5, 1.0])


ax4.set_title('V$_{X}$ (+1/0) Transition Level (eV)', c='k', fontsize=18, pad=7)

#ax4.legend(loc='upper left',ncol=1, frameon=True, prop={'family':'Arial narrow','size':12})
#ax4.savefig('plot_CT4.eps', dpi=450)
#ax4.show()















#plt.tick_params(axis='y', which='both', labelleft=True, labelright=False)

#plt.ylabel('ML Prediction', fontname='Arial Narrow', size=32)
#plt.xlabel('DFT Calculation', fontname='Arial Narrow', size=32)

#plt.rc('xtick', c='k', labelsize=16)
#plt.rc('ytick', c='k', labelsize=24)

plt.savefig('Parity_plot_defect.eps', dpi=450)
plt.show()














