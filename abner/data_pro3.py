import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")
import os
from xgboost import XGBRegressor
from xgboost import plot_importance

#%%load data
items=pd.read_csv("../data/items.csv")
shops=pd.read_csv("../data/shops.csv")
cats=pd.read_csv("../data/item_categories.csv")
train=pd.read_csv("../data/sales_train.csv")
test=pd.read_csv("../data/test.csv")

#%%
'''
plt.figure(figsize=(10,4))
plt.xlim(-100, 3000)
flierprops = dict(marker='o', markerfacecolor='purple', markersize=6,
                  linestyle='none', markeredgecolor='black')
sns.boxplot(x=train.item_cnt_day, flierprops=flierprops)

plt.figure(figsize=(10,4))
plt.xlim(train.item_price.min(), train.item_price.max()*1.1)
sns.boxplot(x=train.item_price, flierprops=flierprops)
'''
#%% Claen
'''
移除單日銷售大於1000或價格大於300000
'''
train = train[(train.item_price < 300000 )& (train.item_cnt_day < 1000)]
train = train[train.item_price > 0].reset_index(drop = True)
train.loc[train.item_cnt_day < 1, "item_cnt_day"] = 0

#%% Duplicates
'''
相同商店合併
'''
train.loc[train.shop_id == 0, 'shop_id'] = 57
test.loc[test.shop_id == 0, 'shop_id'] = 57
train.loc[train.shop_id == 1, 'shop_id'] = 58
test.loc[test.shop_id == 1, 'shop_id'] = 58
train.loc[train.shop_id == 10, 'shop_id'] = 11
test.loc[test.shop_id == 10, 'shop_id'] = 11

#%% Add Shop Loc
'''
加入所在地區
'''
shops.loc[ shops.shop_name == 'Сергиев Посад ТЦ "7Я"',"shop_name" ] = 'СергиевПосад ТЦ "7Я"'
shops["city"] = shops.shop_name.str.split(" ").map( lambda x: x[0] )
shops["category"] = shops.shop_name.str.split(" ").map( lambda x: x[1] )
shops.loc[shops.city == "!Якутск", "city"] = "Якутск"

#%% here are 5 or more shops of that category
'''
共分做6區
'''
category = []
for cat in shops.category.unique():
    if len(shops[shops.category == cat]) >= 5:
        category.append(cat)
shops.category = shops.category.apply( lambda x: x if (x in category) else "other" )

#%%
'''
使用LabelEncoder encode
'''
from sklearn.preprocessing import LabelEncoder
shops["shop_category"] = LabelEncoder().fit_transform( shops.category )
shops["shop_city"] = LabelEncoder().fit_transform( shops.city )
shops = shops[["shop_id", "shop_category", "shop_city"]]

#%% Clean item category
'''
合併類別相同商品
'''
cats["type_code"] = cats.item_category_name.apply( lambda x: x.split(" ")[0] ).astype(str)
cats.loc[ (cats.type_code == "Игровые")| (cats.type_code == "Аксессуары"), "category" ] = "Игры"

#%%
'''
同樣分為5大類並用LabelEncoder encode (64)
'''
category = []
for cat in cats.type_code.unique():
    if len(cats[cats.type_code == cat]) >= 5: 
        category.append(cat)
cats.type_code = cats.type_code.apply(lambda x: x if (x in category) else "etc")

#%%
cats.type_code = LabelEncoder().fit_transform(cats.type_code)
cats["split"] = cats.item_category_name.apply(lambda x: x.split("-"))
cats["subtype"] = cats.split.apply(lambda x: x[1].strip() if len(x) > 1 else x[0].strip())
cats["subtype_code"] = LabelEncoder().fit_transform( cats["subtype"] )
cats = cats[["item_category_id", "subtype_code", "type_code"]]

#%%
data_tra_s = train.sort_values(by=['date_block_num', 'shop_id', 'item_id'])
data_tra = np.array(data_tra_s)[1:,:]
data_tes = np.array(test)
cateID = np.array(items)[:, 1:]
ID = np.array(test)[:, 1:3]

#%%
tes_Z = np.zeros([len(data_tes), 33])
IDD = np.zeros((len(ID), 7))
IDD[:, :2] = ID

# id_m = 20
for ns in range(len(data_tes)):
    print('=====Process >> ' + str(ns) + '/' + str(len(data_tes)) + '=====', "\r", end=' ')
    shop_id = data_tes[ns, 1]
    item_id = data_tes[ns, -1]
    # ====cats====
    cate = cateID[cateID[:,0]==item_id][0,1]
    cats_np = np.array(cats)
    id_c = cats_np[cats_np[:,0]==cate][0, :]
    IDD[ns, 2:5] = id_c
    # ====shops====
    shops_np = np.array(shops)
    id_s = shops_np[shops_np[:,0]==shop_id][0, 1:]
    # ====IDD====
    IDD[ns, 2:5] = id_c
    IDD[ns, 5:7] = id_s
    # ====Train====
    B = data_tra[data_tra[:, 2]==shop_id]
    C = B[B[:, 3]==item_id]  
    for id_m in range(tes_Z.shape[1]):
        A = C[C[:,1]==id_m]
        if A.shape[0]==0:
            tes_Z[ns, id_m] = 0
        else:
            tes_Z[ns, id_m] = np.sum(A[:, -1])  

np.save('tes_Z.npy', tes_Z)
np.save('ID.npy', IDD)