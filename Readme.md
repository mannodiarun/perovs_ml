Regression Example With Boston Dataset: Standardized


```python
from pandas import read_csv
import tensorflow.keras as keras
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
```


```python
import numpy as np     
import csv 
import copy 
import random 
#import mlpy
import pandas
import matplotlib.pyplot as plt 
#from mlpy import KernelRidge                                                                                                                                                     
```


```python
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
X = csvdata[:,13:]
```


```python
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
n = C.size
m = int(X.size/n)
m_y = int(Y.size/n)
#m_y = 4
```


```python
t = 0.20

X_train, X_test, Prop_train, Prop_test, Prop_train_pbe_gap, Prop_test_pbe_gap, Prop_train_hse_gap, Prop_test_hse_gap, Prop_train_ref_ind, Prop_test_ref_ind, Prop_train_fom, Prop_test_fom  =  train_test_split(XX, YY, PBE_gap, HSE_gap, Ref_ind, FOM, test_size=t)

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
        
        
pbe_gap_fl = [0.0]*n
hse_gap_fl = [0.0]*n
ref_ind_fl = [0.0]*n
fom_fl = [0.0]*n

for i in range(0,n):
    pbe_gap_fl[i] = float(PBE_gap[i])
    hse_gap_fl[i] = float(HSE_gap[i])
    ref_ind_fl[i] = float(Ref_ind[i])
    fom_fl[i] = float(FOM[i])
    
max_range_pbe_gap = float ( np.max(pbe_gap_fl[:]) )
min_range_pbe_gap = float ( np.min(pbe_gap_fl[:]) )
max_range_hse_gap = float ( np.max(hse_gap_fl[:]) )
min_range_hse_gap = float ( np.min(hse_gap_fl[:]) )
max_range_ref_ind = float ( np.max(ref_ind_fl[:]) )
min_range_ref_ind = float ( np.min(ref_ind_fl[:]) )
x_range_fom = float ( np.max(fom_fl[:]) )
min_range_fom = float ( np.min(fom_fl[:]) )
```

 NN Optimizers and Model Definition


```python
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
                        
                        keras.optimizers.Adam(lr=lr[a], beta_1=0.9, beta_2=0.999, amsgrad=False)
                        
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
                        
#print(lr[a])

times = 5
```

 Train Model For PBE Band Gap ##


```python
#times = 25
#times = len(pipelines)

train_errors = [0.0]*times
test_errors = [0.0]*times
nn_errors = list()

n_fold = 5
Prop_train = copy.deepcopy(Prop_train_pbe_gap)

for i in range(0,times):
    pipeline = pipelines[np.random.randint(0,729)]
#    pipeline = pipelines[i]
    kf = KFold(n_splits = n_fold)
    mse_test_cv = 0.00
    mse_train_cv = 0.00
    for train, test in kf.split(X_train):
        X_train_cv, X_test_cv, Prop_train_cv, Prop_test_cv = X_train[train], X_train[test], Prop_train[train], Prop_train[test]
        pipeline.fit(X_train_cv,Prop_train_cv)
        Prop_pred_train_cv = pipeline.predict(X_train_cv)
        Prop_pred_test_cv  = pipeline.predict(X_test_cv)
        n_te_cv = Prop_test_cv.size
        n_tr_cv = Prop_train_cv.size
        Prop_test_cv_fl = [0.0]*n_te_cv
        Prop_pred_test_cv_fl = [0.0]*n_te_cv
        for aa in range(0,n_te_cv):
            Prop_test_cv_fl[aa] = np.float(Prop_test_cv[aa])
            Prop_pred_test_cv_fl[aa] = np.float(Prop_pred_test_cv[aa])
        Prop_train_cv_fl = [0.0]*n_tr_cv
        Prop_pred_train_cv_fl = [0.0]*n_tr_cv
        for aa in range(0,n_tr_cv):
            Prop_train_cv_fl[aa] = np.float(Prop_train_cv[aa])
            Prop_pred_train_cv_fl[aa] = np.float(Prop_pred_train_cv[aa])
        mse_test_cv = mse_test_cv  + sklearn.metrics.mean_squared_error(Prop_test_cv_fl, Prop_pred_test_cv_fl)
        mse_train_cv = mse_train_cv + sklearn.metrics.mean_squared_error(Prop_train_cv_fl, Prop_pred_train_cv_fl)
    mse_test = mse_test_cv / n_fold
    mse_train = mse_train_cv / n_fold
    train_errors[i] = mse_train
    test_errors[i] = mse_test
    nn_errors.append(pipeline)
i_opt = np.argmin(test_errors)
pipeline_opt = nn_errors[i_opt]

train_errors_pbe_gap = copy.deepcopy(train_errors)
test_errors_pbe_gap  = copy.deepcopy(test_errors)

pipeline_opt.fit(X_train, Prop_train)
Pred_train = pipeline_opt.predict(X_train)
Pred_test  = pipeline_opt.predict(X_test)

Pred_train_pbe_gap_fl = [0.0]*n_tr
Pred_test_pbe_gap_fl  = [0.0]*n_te
Prop_train_pbe_gap_fl = [0.0]*n_tr
Prop_test_pbe_gap_fl  = [0.0]*n_te
for i in range(0,n_tr):
    Pred_train_pbe_gap_fl[i] = np.float(Pred_train[i])
    Prop_train_pbe_gap_fl[i] = np.float(Prop_train_pbe_gap[i])
for i in range(0,n_te):
    Pred_test_pbe_gap_fl[i] = np.float(Pred_test[i])
    Prop_test_pbe_gap_fl[i] = np.float(Prop_test_pbe_gap[i])
    
    
## Outside Predictions ##

Pred_out = pipeline_opt.predict(X_out)
Pred_out_pbe_gap = [0.0]*n_out
for i in range(0,n_out):
    Pred_out_pbe_gap[i] = np.float(Pred_out[i])
```

 Train Model For HSE Band Gap ##


```python
#times = 25
#times = len(pipelines)

train_errors = [0.0]*times
test_errors = [0.0]*times
nn_errors = list()

n_fold = 5
Prop_train = copy.deepcopy(Prop_train_hse_gap)

for i in range(0,times):
    pipeline = pipelines[np.random.randint(0,729)]
#    pipeline = pipelines[i]
    kf = KFold(n_splits = n_fold)
    mse_test_cv = 0.00
    mse_train_cv = 0.00
    for train, test in kf.split(X_train):
        X_train_cv, X_test_cv, Prop_train_cv, Prop_test_cv = X_train[train], X_train[test], Prop_train[train], Prop_train[test]
        pipeline.fit(X_train_cv,Prop_train_cv)
        Prop_pred_train_cv = pipeline.predict(X_train_cv)
        Prop_pred_test_cv  = pipeline.predict(X_test_cv)
        n_te_cv = Prop_test_cv.size
        n_tr_cv = Prop_train_cv.size
        Prop_test_cv_fl = [0.0]*n_te_cv
        Prop_pred_test_cv_fl = [0.0]*n_te_cv
        for aa in range(0,n_te_cv):
            Prop_test_cv_fl[aa] = np.float(Prop_test_cv[aa])
            Prop_pred_test_cv_fl[aa] = np.float(Prop_pred_test_cv[aa])
        Prop_train_cv_fl = [0.0]*n_tr_cv
        Prop_pred_train_cv_fl = [0.0]*n_tr_cv
        for aa in range(0,n_tr_cv):
            Prop_train_cv_fl[aa] = np.float(Prop_train_cv[aa])
            Prop_pred_train_cv_fl[aa] = np.float(Prop_pred_train_cv[aa])
        mse_test_cv = mse_test_cv  + sklearn.metrics.mean_squared_error(Prop_test_cv_fl, Prop_pred_test_cv_fl)
        mse_train_cv = mse_train_cv + sklearn.metrics.mean_squared_error(Prop_train_cv_fl, Prop_pred_train_cv_fl)
    mse_test = mse_test_cv / n_fold
    mse_train = mse_train_cv / n_fold
    train_errors[i] = mse_train
    test_errors[i] = mse_test
    nn_errors.append(pipeline)
i_opt = np.argmin(test_errors)
pipeline_opt = nn_errors[i_opt]

train_errors_hse_gap = copy.deepcopy(train_errors)
test_errors_hse_gap  = copy.deepcopy(test_errors)

pipeline_opt.fit(X_train,Prop_train)
Pred_train = pipeline_opt.predict(X_train)
Pred_test  = pipeline_opt.predict(X_test)

Pred_train_hse_gap_fl = [0.0]*n_tr
Pred_test_hse_gap_fl  = [0.0]*n_te
Prop_train_hse_gap_fl = [0.0]*n_tr
Prop_test_hse_gap_fl  = [0.0]*n_te
for i in range(0,n_tr):
    Pred_train_hse_gap_fl[i] = np.float(Pred_train[i])
    Prop_train_hse_gap_fl[i] = np.float(Prop_train_hse_gap[i])
for i in range(0,n_te):
    Pred_test_hse_gap_fl[i] = np.float(Pred_test[i])
    Prop_test_hse_gap_fl[i] = np.float(Prop_test_hse_gap[i])
    
    
## Outside Predictions ##

Pred_out = pipeline_opt.predict(X_out)
Pred_out_hse_gap = [0.0]*n_out
for i in range(0,n_out):
    Pred_out_hse_gap[i] = np.float(Pred_out[i])
```

 Train Model For Refractive Index ##


```python
#times = 25
#times = len(pipelines)

train_errors = [0.0]*times
test_errors = [0.0]*times
nn_errors = list()

n_fold = 5
Prop_train = copy.deepcopy(Prop_train_ref_ind)

for i in range(0,times):
    pipeline = pipelines[np.random.randint(0,729)]
#    pipeline = pipelines[i]
    kf = KFold(n_splits = n_fold)
    mse_test_cv = 0.00
    mse_train_cv = 0.00
    for train, test in kf.split(X_train):
        X_train_cv, X_test_cv, Prop_train_cv, Prop_test_cv = X_train[train], X_train[test], Prop_train[train], Prop_train[test]
        pipeline.fit(X_train_cv,Prop_train_cv)
        Prop_pred_train_cv = pipeline.predict(X_train_cv)
        Prop_pred_test_cv  = pipeline.predict(X_test_cv)
        n_te_cv = Prop_test_cv.size
        n_tr_cv = Prop_train_cv.size
        Prop_test_cv_fl = [0.0]*n_te_cv
        Prop_pred_test_cv_fl = [0.0]*n_te_cv
        for aa in range(0,n_te_cv):
            Prop_test_cv_fl[aa] = np.float(Prop_test_cv[aa])
            Prop_pred_test_cv_fl[aa] = np.float(Prop_pred_test_cv[aa])
        Prop_train_cv_fl = [0.0]*n_tr_cv
        Prop_pred_train_cv_fl = [0.0]*n_tr_cv
        for aa in range(0,n_tr_cv):
            Prop_train_cv_fl[aa] = np.float(Prop_train_cv[aa])
            Prop_pred_train_cv_fl[aa] = np.float(Prop_pred_train_cv[aa])
        mse_test_cv = mse_test_cv  + sklearn.metrics.mean_squared_error(Prop_test_cv_fl, Prop_pred_test_cv_fl)
        mse_train_cv = mse_train_cv + sklearn.metrics.mean_squared_error(Prop_train_cv_fl, Prop_pred_train_cv_fl)
    mse_test = mse_test_cv / n_fold
    mse_train = mse_train_cv / n_fold
    train_errors[i] = mse_train
    test_errors[i] = mse_test
    nn_errors.append(pipeline)
i_opt = np.argmin(test_errors)
pipeline_opt = nn_errors[i_opt]

train_errors_ref_ind = copy.deepcopy(train_errors)
test_errors_ref_ind  = copy.deepcopy(test_errors)

pipeline_opt.fit(X_train, Prop_train)
Pred_train = pipeline_opt.predict(X_train)
Pred_test  = pipeline_opt.predict(X_test)

Pred_train_ref_ind_fl = [0.0]*n_tr
Pred_test_ref_ind_fl  = [0.0]*n_te
Prop_train_ref_ind_fl = [0.0]*n_tr
Prop_test_ref_ind_fl  = [0.0]*n_te
for i in range(0,n_tr):
    Pred_train_ref_ind_fl[i] = np.float(Pred_train[i])
    Prop_train_ref_ind_fl[i] = np.float(Prop_train_ref_ind[i])
for i in range(0,n_te):
    Pred_test_ref_ind_fl[i] = np.float(Pred_test[i])
    Prop_test_ref_ind_fl[i] = np.float(Prop_test_ref_ind[i])
    
    
## Outside Predictions ##

Pred_out = pipeline_opt.predict(X_out)
Pred_out_ref_ind = [0.0]*n_out
for i in range(0,n_out):
    Pred_out_ref_ind[i] = np.float(Pred_out[i])
```

 Train Model For Figure of Merit ##


```python
#times = 25
#times = len(pipelines)

train_errors = [0.0]*times
test_errors = [0.0]*times
nn_errors = list()

n_fold = 5
Prop_train = copy.deepcopy(Prop_train_fom)

for i in range(0,times):
    pipeline = pipelines[np.random.randint(0,729)]
#    pipeline = pipelines[i]
    kf = KFold(n_splits = n_fold)
    mse_test_cv = 0.00
    mse_train_cv = 0.00
    for train, test in kf.split(X_train):
        X_train_cv, X_test_cv, Prop_train_cv, Prop_test_cv = X_train[train], X_train[test], Prop_train[train], Prop_train[test]
        pipeline.fit(X_train_cv,Prop_train_cv)
        Prop_pred_train_cv = pipeline.predict(X_train_cv)
        Prop_pred_test_cv  = pipeline.predict(X_test_cv)
        n_te_cv = Prop_test_cv.size
        n_tr_cv = Prop_train_cv.size
        Prop_test_cv_fl = [0.0]*n_te_cv
        Prop_pred_test_cv_fl = [0.0]*n_te_cv
        for aa in range(0,n_te_cv):
            Prop_test_cv_fl[aa] = np.float(Prop_test_cv[aa])
            Prop_pred_test_cv_fl[aa] = np.float(Prop_pred_test_cv[aa])
        Prop_train_cv_fl = [0.0]*n_tr_cv
        Prop_pred_train_cv_fl = [0.0]*n_tr_cv
        for aa in range(0,n_tr_cv):
            Prop_train_cv_fl[aa] = np.float(Prop_train_cv[aa])
            Prop_pred_train_cv_fl[aa] = np.float(Prop_pred_train_cv[aa])
        mse_test_cv = mse_test_cv  + sklearn.metrics.mean_squared_error(Prop_test_cv_fl, Prop_pred_test_cv_fl)
        mse_train_cv = mse_train_cv + sklearn.metrics.mean_squared_error(Prop_train_cv_fl, Prop_pred_train_cv_fl)
    mse_test = mse_test_cv / n_fold
    mse_train = mse_train_cv / n_fold
    train_errors[i] = mse_train
    test_errors[i] = mse_test
    nn_errors.append(pipeline)
i_opt = np.argmin(test_errors)
pipeline_opt = nn_errors[i_opt]

train_errors_fom = copy.deepcopy(train_errors)
test_errors_fom  = copy.deepcopy(test_errors)

pipeline_opt.fit(X_train,Prop_train)
Pred_train = pipeline_opt.predict(X_train)
Pred_test  = pipeline_opt.predict(X_test)

Pred_train_fom_fl = [0.0]*n_tr
Pred_test_fom_fl  = [0.0]*n_te
Prop_train_fom_fl = [0.0]*n_tr
Prop_test_fom_fl  = [0.0]*n_te
for i in range(0,n_tr):
    Pred_train_fom_fl[i] = np.float(Pred_train[i])
    Prop_train_fom_fl[i] = np.float(Prop_train_fom[i])
for i in range(0,n_te):
    Pred_test_fom_fl[i] = np.float(Pred_test[i])
    Prop_test_fom_fl[i] = np.float(Prop_test_fom[i])
    
    
## Outside Predictions ##

Pred_out = pipeline_opt.predict(X_out)
Pred_out_fom = [0.0]*n_out
for i in range(0,n_out):
    Pred_out_fom[i] = np.float(Pred_out[i])

```


```python
errors = [[0.0 for a in range(8)] for b in range(times)]

for i in range(0,times):
    errors[i][0] = train_errors_pbe_gap[i]
    errors[i][1] = test_errors_pbe_gap[i]
    errors[i][2] = train_errors_hse_gap[i]
    errors[i][3] = test_errors_hse_gap[i]
    errors[i][4] = train_errors_ref_ind[i]
    errors[i][5] = test_errors_ref_ind[i]
    errors[i][6] = train_errors_fom[i]
    errors[i][7] = test_errors_fom[i]
    
np.savetxt('errors.txt', errors)

```


```python
Pred_out = [[0.0 for a in range(4)] for b in range(n_out)]

for i in range(0,n_out):
    Pred_out[i][0] = Pred_out_pbe_gap[i]
    Pred_out[i][1] = Pred_out_hse_gap[i]
    Pred_out[i][2] = Pred_out_ref_ind[i]
    Pred_out[i][3] = Pred_out_fom[i]
    
np.savetxt('Pred_out.txt', Pred_out)
```


```python
mse_test_prop  = sklearn.metrics.mean_squared_error(Prop_test_pbe_gap_fl, Pred_test_pbe_gap_fl)
mse_train_prop = sklearn.metrics.mean_squared_error(Prop_train_pbe_gap_fl, Pred_train_pbe_gap_fl)
rmse_test_pbe_gap  = np.sqrt(mse_test_prop)
rmse_train_pbe_gap = np.sqrt(mse_train_prop)
print('rmse_test_pbe_gap = ', np.sqrt(mse_test_prop))
print('rmse_train_pbe_gap = ', np.sqrt(mse_train_prop))
print('      ')

mse_test_prop  = sklearn.metrics.mean_squared_error(Prop_test_hse_gap_fl, Pred_test_hse_gap_fl)
mse_train_prop = sklearn.metrics.mean_squared_error(Prop_train_hse_gap_fl, Pred_train_hse_gap_fl)
rmse_test_hse_gap  = np.sqrt(mse_test_prop)
rmse_train_hse_gap = np.sqrt(mse_train_prop)
print('rmse_test_hse_gap = ', np.sqrt(mse_test_prop))
print('rmse_train_hse_gap = ', np.sqrt(mse_train_prop))
print('      ')

mse_test_prop  = sklearn.metrics.mean_squared_error(Prop_test_ref_ind_fl, Pred_test_ref_ind_fl)
mse_train_prop = sklearn.metrics.mean_squared_error(Prop_train_ref_ind_fl, Pred_train_ref_ind_fl)
rmse_test_ref_ind  = np.sqrt(mse_test_prop)
rmse_train_ref_ind = np.sqrt(mse_train_prop)
print('rmse_test_ref_ind = ', np.sqrt(mse_test_prop))
print('rmse_train_ref_ind = ', np.sqrt(mse_train_prop))
print('      ')

mse_test_prop  = sklearn.metrics.mean_squared_error(Prop_test_fom_fl, Pred_test_fom_fl)
mse_train_prop = sklearn.metrics.mean_squared_error(Prop_train_fom_fl, Pred_train_fom_fl)
rmse_test_fom  = np.sqrt(mse_test_prop)
rmse_train_fom = np.sqrt(mse_train_prop)
print('rmse_test_fom = ', np.sqrt(mse_test_prop))
print('rmse_train_fom = ', np.sqrt(mse_train_prop))
print('      ')
```

 ML Parity Plots ##


```python
fig, ( [ax1, ax2], [ax3, ax4] ) = plt.subplots( nrows=2, ncols=2, figsize=(8,8) )

fig.text(0.5, 0.02, 'DFT Calculation', ha='center', fontsize=32)
fig.text(0.01, 0.5, 'ML Prediction', va='center', rotation='vertical', fontsize=32)

plt.subplots_adjust(left=0.12, bottom=0.12, right=0.97, top=0.94, wspace=0.3, hspace=0.35)
plt.rc('font', family='Arial narrow')
#plt.tight_layout()
#plt.tight_layout(pad=0.6, w_pad=0.5, h_pad=0.5)



Prop_train_temp = copy.deepcopy(Prop_train_pbe_gap_fl)
Pred_train_temp = copy.deepcopy(Pred_train_pbe_gap_fl)
Prop_test_temp  = copy.deepcopy(Prop_test_pbe_gap_fl)
Pred_test_temp  = copy.deepcopy(Pred_test_pbe_gap_fl)
a = [-175,0,125]
b = [-175,0,125]
ax1.plot(b, a, c='k', ls='-')
ax1.xaxis.set_tick_params(labelsize=20)
ax1.yaxis.set_tick_params(labelsize=20)
ax1.scatter(Prop_train_temp[:], Pred_train_temp[:], c='blue', marker='s', s=60, edgecolors='dimgrey', alpha=1.0, label='Training')
ax1.scatter(Prop_test_temp[:], Pred_test_temp[:], c='orange', marker='s', s=60, edgecolors='dimgrey', alpha=0.2, label='Test')
te = '%.2f' % rmse_test_pbe_gap
tr = '%.2f' % rmse_train_pbe_gap
ax1.text(2.95, 0.8, 'Test_rmse = ', c='r', fontsize=12)
ax1.text(4.61, 0.8, te, c='r', fontsize=12)
ax1.text(5.22, 0.8, 'eV', c='r', fontsize=12)
ax1.text(2.84, 0.32, 'Train_rmse = ', c='r', fontsize=12)
ax1.text(4.60, 0.32, tr, c='r', fontsize=12)
ax1.text(5.22, 0.32, 'eV', c='r', fontsize=12)
ax1.set_ylim([-0.2, 5.8])
ax1.set_xlim([-0.2, 5.8])
ax1.set_xticks([1, 2, 3, 4, 5])
ax1.set_yticks([1, 2, 3, 4, 5])
ax1.set_title('PBE Band Gap (eV)', c='k', fontsize=20, pad=8)
ax1.legend(loc='upper left',ncol=1, frameon=True, prop={'family':'Arial narrow','size':12})


Prop_train_temp = copy.deepcopy(Prop_train_hse_gap_fl)
Pred_train_temp = copy.deepcopy(Pred_train_hse_gap_fl)
Prop_test_temp  = copy.deepcopy(Prop_test_hse_gap_fl)
Pred_test_temp  = copy.deepcopy(Pred_test_hse_gap_fl)
ax2.xaxis.set_tick_params(labelsize=20)
ax2.yaxis.set_tick_params(labelsize=20)
ax2.scatter(Prop_train_temp[:], Pred_train_temp[:], c='blue', marker='s', s=60, edgecolors='dimgrey', alpha=1.0, label='Training')
ax2.scatter(Prop_test_temp[:], Pred_test_temp[:], c='orange', marker='s', s=60, edgecolors='dimgrey', alpha=0.2, label='Test')
te = '%.2f' % rmse_test_hse_gap
tr = '%.2f' % rmse_train_hse_gap
ax2.text(4.03, 1.65, 'Test_rmse = ', c='r', fontsize=12)
ax2.text(5.95, 1.65, te, c='r', fontsize=12)
ax2.text(6.67, 1.65, 'eV', c='r', fontsize=12)
ax2.text(3.94, 1.10, 'Train_rmse = ', c='r', fontsize=12)
ax2.text(5.95, 1.10, tr, c='r', fontsize=12)
ax2.text(6.67, 1.10, 'eV', c='r', fontsize=12)
ax2.set_ylim([0.5, 7.3])
ax2.set_xlim([0.5, 7.3])
ax2.set_xticks([1, 3, 5, 7])
ax2.set_yticks([1, 3, 5, 7])
ax2.set_title('HSE Band Gap (eV)', c='k', fontsize=20, pad=8)


Prop_train_temp = copy.deepcopy(Prop_train_ref_ind_fl)
Pred_train_temp = copy.deepcopy(Pred_train_ref_ind_fl)
Prop_test_temp  = copy.deepcopy(Prop_test_ref_ind_fl)
Pred_test_temp  = copy.deepcopy(Pred_test_ref_ind_fl)
ax3.xaxis.set_tick_params(labelsize=20)
ax3.yaxis.set_tick_params(labelsize=20)
ax3.scatter(Prop_train_temp[:], Pred_train_temp[:], c='blue', marker='s', s=60, edgecolors='dimgrey', alpha=1.0, label='Training')
ax3.scatter(Prop_test_temp[:], Pred_test_temp[:], c='orange', marker='s', s=60, edgecolors='dimgrey', alpha=0.2, label='Test')
te = '%.2f' % rmse_test_ref_ind
tr = '%.2f' % rmse_train_ref_ind
ax3.text(2.30, 1.6, 'Test_rmse = ', c='r', fontsize=12)
ax3.text(2.85, 1.6, te, c='r', fontsize=12)
ax3.text(3.05, 1.6, 'eV', c='r', fontsize=12)
ax3.text(2.28, 1.43, 'Train_rmse = ', c='r', fontsize=12)
ax3.text(2.85, 1.43, tr, c='r', fontsize=12)
ax3.text(3.05, 1.43, 'eV', c='r', fontsize=12)
ax3.set_ylim([1.25, 3.25])
ax3.set_xlim([1.25, 3.25])
ax3.set_xticks([1.5, 2.0, 2.5, 3.0])
ax3.set_yticks([1.5, 2.0, 2.5, 3.0])
ax3.set_title('Refractive Index', c='k', fontsize=20, pad=8)


Prop_train_temp = copy.deepcopy(Prop_train_fom_fl)
Pred_train_temp = copy.deepcopy(Pred_train_fom_fl)
Prop_test_temp  = copy.deepcopy(Prop_test_fom_fl)
Pred_test_temp  = copy.deepcopy(Pred_test_fom_fl)
ax4.xaxis.set_tick_params(labelsize=20)
ax4.yaxis.set_tick_params(labelsize=20)
ax4.scatter(Prop_train_temp[:], Pred_train_temp[:], c='blue', marker='s', s=60, edgecolors='dimgrey', alpha=1.0, label='Training')
ax4.scatter(Prop_test_temp[:], Pred_test_temp[:], c='orange', marker='s', s=60, edgecolors='dimgrey', alpha=0.2, label='Test')
te = '%.2f' % rmse_test_fom
tr = '%.2f' % rmse_train_fom
ax4.text(3.61, 2.2, 'Test_rmse = ', c='r', fontsize=12)
ax4.text(4.7, 2.2, te, c='r', fontsize=12)
ax4.text(5.1, 2.2, 'eV', c='r', fontsize=12)
ax4.text(3.55, 1.85, 'Train_rmse = ', c='r', fontsize=12)
ax4.text(4.7, 1.85, tr, c='r', fontsize=12)
ax4.text(5.1, 1.85, 'eV', c='r', fontsize=12)
ax4.set_ylim([1.5, 5.5])
ax4.set_xlim([1.5, 5.5])
ax4.set_xticks([2.0, 3.0, 4.0, 5.0])
ax4.set_yticks([2.0, 3.0, 4.0, 5.0])
ax4.set_title('Figure of Merit (log$_{10}$)', c='k', fontsize=20, pad=8)


plt.savefig('plot.eps', dpi=450)
plt.show()

```
