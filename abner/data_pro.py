import numpy as np
import pandas as pd
import os

path = '../data'
Ftra = 'sales_train.csv'
Data_tra = pd.read_csv(os.path.join(path, Ftra), low_memory=False)
data_tra_s = Data_tra.sort_values(by=['date_block_num', 'shop_id', 'item_id'])
data_tra = np.array(data_tra_s)[1:,:]

id_m = 0
RES_m = []
for id_m in range(34):
    print('month >> ', str(id_m))
    A = data_tra[data_tra[:,1]==str(id_m)]
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
