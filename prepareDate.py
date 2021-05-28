import pandas as pd
from datetime import datetime
import numpy as np
from itertools import product

def get_train_data():
    # READ
    df = pd.read_csv('./data/sales_train.csv')
    df_itemCategory = pd.read_csv('./data/items.csv')

    # PROCCESS
    df['date'] = [datetime.strptime(date_str, "%d.%m.%Y") for date_str in df['date'].values]

    item_cate = {}
    for i in df_itemCategory.index:
        item_cate[df_itemCategory['item_id'][i]] = df_itemCategory['item_category_id'][i]
    df['item_category_id'] = [item_cate[item_id] for item_id in df['item_id']]

    # COUNT
    month_cnt = np.unique(df['date_block_num']).shape[0]
    shop_cnt = np.unique(df['shop_id']).shape[0]
    cate_cnt = np.unique(df['item_category_id']).shape[0]
    item_cnt = np.unique(df_itemCategory['item_id']).shape[0]

    #
    # [month,shop,item,(count,category)]
    d = np.zeros([month_cnt,shop_cnt,item_cnt,2])

    shop_id = df['shop_id']
    item_id = df['item_id']
    month_id = df['date_block_num']
    item_id_cnt = df['item_cnt_day']

    for month in range(month_cnt):
        for shop in range(shop_cnt):        
            select_item_id = item_id[(month_id == month) & (shop_id == shop)]
            select_item_cnt = item_id_cnt[(month_id == month) & (shop_id == shop)]
                    
            d[month,shop,select_item_id,0] += select_item_cnt   

    d[:,:,:,1] = df_itemCategory['item_category_id']


    # WINDOW
    data_input = np.zeros([item_cnt*shop_cnt,month_cnt,4])
    wn = np.blackman(8)

    for n,(shop,item) in enumerate(product(range(shop_cnt),range(item_cnt))):    
        data_input[n,:,0] = np.convolve(d[:,shop,item,0],wn,mode='same')   

        data_input[n,:,1] = d[:,shop,item,1] 

        data_input[n,:,2] = np.repeat(shop, month_cnt)
        data_input[n,:,3] = np.repeat(item, month_cnt)

    # TRAINING DATA
    SEQ_LENGTH = 12
    N, TIME_LENGTH, FEATURE_LENGTH = data_input.shape

    x, y = list(), list()
    for i in range(0,TIME_LENGTH - (SEQ_LENGTH + 1) + 1, 1):    
        slice_data = data_input[:,i:i+SEQ_LENGTH+1,:]
        
        x.append(slice_data[:,:-1,:])
        y.append(slice_data[:,-1:,0]) 

    train_x = np.vstack(x)
    train_y = np.vstack(y)
    del x,y

    return train_x, train_y

get_train_data()