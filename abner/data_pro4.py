import numpy as np 


#%%load data
x = np.load('inputs_ref.npz')

# def _inputs(x):
X_train = (x['X_train'])
Y_train = (x['Y_train'])
X_valid = (x['X_valid'])
Y_valid = (x['Y_valid'])
X_test = (x['X_test'])

TRAIN = np.vstack((X_train, X_valid))
LABEL = np.vstack((Y_train[:, np.newaxis], Y_valid[:, np.newaxis]))

DATA = np.hstack((TRAIN, LABEL))


tra = []
for i in range(len(X_test)):
    print('=====Process >> ' + str(i) + '/' + str(len(X_test)) + '=====', "\r", end=' ')
    shop_idx = X_test[i,1]
    item_idx = X_test[i,2]
    A = DATA[DATA[:,1]==shop_idx]
    B = A[A[:,2]==item_idx]
    if len(B)!=0:
        tra.append(B)
#%%        
tra = np.vstack(tra)
X = tra[:, :-1]
Y = tra[:, -1]

np.savez_compressed('inputs_ref_onlyTES.npz', X_train=X, Y_train=Y, X_test=X_test)