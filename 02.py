import pandas as pd
import numpy as np
import h5py


#%%
df = pd.read_hdf('preprocessData.h5', key='df', mode='r')

print(df.keys())


train_df = df[df['date_block_num'] < 34]
test_df = df[df['date_block_num'] == 34]


def createInput(df: pd.DataFrame):
    emb_features = list()

    # need EMB
    emb_cols = ['month', 'days', 'shop_id', 'item_id', 'shopcategory_id', 'shop_city_id',
                'item_category_id', 'item_type_1_id', 'item_type_2_id', 'cate_type_id', 'cate_subtype_id']

    for col in emb_cols:
        feature = np.vstack(df[col])
        emb_features.append(feature)

    emb_features = np.concatenate(emb_features, axis=-1)

    # TIME
    other_feature = list()
    del feature

    feature = np.vstack(df['date_avg_item_cnt_lag_1'])
    other_feature.append(feature)

    feature = np.vstack(df['delta_price_lag'])
    other_feature.append(feature)

    other_feature = np.concatenate(other_feature, axis=-1)

    #
    time_feature = list()
    del feature

    feature = [np.expand_dims(
        df[f'date_item_avg_item_cnt_lag_{c+1}'], axis=1) for c in range(3)]
    feature = np.concatenate(feature, axis=1)
    feature = np.expand_dims(feature, axis=2)
    time_feature.append(feature)

    feature = [np.expand_dims(
        df[f'date_shop_avg_item_cnt_lag_{c+1}'], axis=1) for c in range(3)]
    feature = np.concatenate(feature, axis=1)
    feature = np.expand_dims(feature, axis=2)
    time_feature.append(feature)

    time_feature = np.concatenate(time_feature, axis=-1)

    #
    
    del feature

    feature = [np.expand_dims(
        df[f'item_cnt_month_lag_{c+1}'], axis=1) for c in range(12)]
    feature = np.concatenate(feature, axis=1)
    feature = np.expand_dims(feature, axis=2)


    return (emb_features, other_feature, time_feature, feature)


def createLabel(df: pd.DataFrame):
    item_cnt_month = np.vstack(df['item_cnt_month'])
    return item_cnt_month


train_x = createInput(train_df)
train_y = createLabel(train_df)
test_x = createInput(test_df)

'''
np.savez_compressed('inputs.npz', train_x_emb=train_x[0], train_x_other=train_x[1],
                    train_x_time=train_x[2], train_x_cnt=train_x[3], train_y=train_y,  test_x_emb=test_x[0], 
                    test_x_other=test_x[1], test_x_time=test_x[2], test_x_cnt=test_x[3])
'''


