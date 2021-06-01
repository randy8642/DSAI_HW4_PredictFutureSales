import numpy as np
import pandas as pd
import os

path = '../data'
Ftra = 'sales_train.csv'
Ftes = 'test.csv'
Data_tra = pd.read_csv(os.path.join(path, Ftra), low_memory=False)
Data_tes = pd.read_csv(os.path.join(path, Ftes), low_memory=False)
data_tra_s = Data_tra.sort_values(by=['date_block_num', 'shop_id', 'item_id'])
data_tra = np.array(data_tra_s)[1:,:]
data_tes = np.array(Data_tes)

#%%
tes_Z = np.zeros([len(data_tes), 33])
# id_m = 20
for ns in range(len(data_tes)):
    shop_id = data_tes[ns, 1]
    item_id = data_tes[ns, -1]
    B = data_tra[data_tra[:, 2]==shop_id]
    C = B[B[:, 3]==item_id]  
    for id_m in range(tes_Z.shape[1]):
        A = C[C[:,1]==id_m]
        if A.shape[0]==0:
            tes_Z[ns, id_m] = 0
        else:
            tes_Z[ns, id_m] = np.sum(A[:, -1])        
'''
id_m = 0
RES_m = []
for id_m in range(34):
    print('month >> ', str(id_m))
    A = data_tra[data_tra[:,1]==id_m]
    #%%
    shop_id = '666'
    item_id = '1019'
    RES = []
    for i in range(len(A)):
        if A[i, 2]!=shop_id or A[i, 3]!=item_id:
            shop_id = A[i, 2]
            item_id = A[i, 3]
            B = A[A[:, 2]==shop_id]
            C = B[B[:, 3]==item_id]
            tot_cnt = np.sum(np.array(C[:, 1:], dtype=np.float), axis=0)[-1]
            res = np.array([id_m, shop_id, item_id, tot_cnt], dtype=np.float)[np.newaxis, :]
            RES.append(res)
    res = np.vstack(RES)
    RES_m.append(res)
res_m = np.vstack(RES_m)
np.save('res.npy', res_m)
'''
