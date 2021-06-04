import pandas as pd
import numpy as np
from itertools import product
import matplotlib.pyplot as plt
import copy
import re


def train_preprocess(df: pd.DataFrame):
    # 移除離群值
    df = df[(df['item_price'] < 300000) & (df['item_cnt_day'] < 1000)]
    df = df[df['item_price'] > 0].reset_index(drop=True)
    df.loc[df['item_cnt_day'] < 1, 'item_cnt_day'] = 0

    # 合併相同商店(名稱相同)
    same_shopId = [[0, 57], [1, 58], [10, 11]]
    for pair in same_shopId:
        df.loc[df['shop_id'] == pair[0], 'shop_id'] = pair[1]

    return df


def test_preprocess(df: pd.DataFrame):
    # 合併相同商店(名稱相同)
    same_shopId = [[0, 57], [1, 58], [10, 11]]
    for pair in same_shopId:
        df.loc[df['shop_id'] == pair[0], 'shop_id'] = pair[1]

    return df


def shop_preprocess(df: pd.DataFrame):

    # 切割商店名稱 => 所在地 & 類型
    df["city"] = df['shop_name'].str.split(
        ' ').map(lambda x: ' '.join(x[0:-2]))
    df["category"] = df['shop_name'].str.split(' ').map(lambda x: x[-2])

    df["city"] = df["city"].apply(lambda x: re.sub('[!]', '', x))

    # 合併過小的類別
    category = []
    for cat in df['category'].unique():
        if len(df[df['category'] == cat]) >= 5:
            category.append(cat)
    df['category'] = df['category'].apply(
        lambda x: x if (x in category) else "other")

    # 標籤(文字) => 索引(類別編號)
    _, df['shopcategory_id'] = np.unique(
        df['category'], return_inverse=True)
    _, df['shop_city_id'] = np.unique(
        df['city'], return_inverse=True)

    # 新df
    df = df[["shop_id", "shopcategory_id", "shop_city_id"]]

    return df


def category_preprocess(df: pd.DataFrame):

    # 切割類別名稱 => 主類別 & 子類別
    df['cate_type'] = df['item_category_name'].apply(
        lambda x: re.split(' - ', x)[0]).astype(str)
    df['cate_subtype'] = df['item_category_name'].apply(
        lambda x: re.split(' - ', x)[-1]).astype(str)

    # 合併過小的類別
    category = []
    for cat in df['cate_type'].unique():
        if len(df[df['cate_type'] == cat]) >= 5:
            category.append(cat)
    df['cate_type'] = df['cate_type'].apply(
        lambda x: x if (x in category) else 'etc')

    # 標籤(文字) => 索引(類別編號)
    _, df['cate_type_id'] = np.unique(df['cate_type'], return_inverse=True)
    _, df['cate_subtype_id'] = np.unique(
        df['cate_subtype'], return_inverse=True)

    # 新df
    df = df[['item_category_id', 'cate_type_id', 'cate_subtype_id']]

    return df


def items_preprocess(df: pd.DataFrame):

    # 切割商品名稱 => 類別1 & 類別2
    def splitNameType1(x):
        result = re.findall(r'\(.*\)', x)
        for index in range(len(result)):
            result[index] = re.sub(',', '', result[index][1:-1])

        if len(result) == 0:
            return x

        return ' '.join(result)

    def splitNameType2(x):
        result = re.findall(r'\[.*\]', x)
        for index in range(len(result)):
            result[index] = re.sub(',', '', result[index][1:-1])

        if len(result) == 0:
            return x

        return ' '.join(result)

    df['item_type_1'] = df['item_name'].apply(lambda x: splitNameType1(x))
    df['item_type_2'] = df['item_name'].apply(lambda x: splitNameType2(x))

    # 整理名稱 並 轉換成小寫字母
    df['item_type_1'] = df['item_type_1'].str.replace(
        '[^A-Za-z0-9А-Яа-я]+', " ").str.lower()
    df['item_type_2'] = df['item_type_2'].str.replace(
        '[^A-Za-z0-9А-Яа-я]+', " ").str.lower()

    # 合併過小的類別
    def cleanSmalltypes(_df: pd.DataFrame, col: str):
        item_type = []
        for item in _df[col].unique():
            if len(_df[_df[col] == item]) >= 40:
                item_type.append(item)
        return _df[col].apply(lambda x: x if (x in item_type) else 'other')

    df['item_type_1'] = cleanSmalltypes(df, 'item_type_1')
    df['item_type_2'] = cleanSmalltypes(df, 'item_type_2')

    # 標籤(文字) => 索引(類別編號)
    _, df['item_type_1_id'] = np.unique(df['item_type_1'], return_inverse=True)
    _, df['item_type_2_id'] = np.unique(df['item_type_2'], return_inverse=True)

    # 新df
    df = df[['item_id', 'item_category_id', 'item_type_1_id', 'item_type_2_id']]

    return df


def save_dataframe(df: pd.DataFrame):
    df.to_hdf('preprocessData.h5', key='df', mode='w', complevel=9)


def createTotalDataframe():
    matrix = []
    cols  = ["date_block_num", "shop_id", "item_id"]
    for i in range(34):
        sales = df_train[df_train.date_block_num == i]
        matrix.append( np.array(list( product( [i], sales['shop_id'].unique(), sales['item_id'].unique() ) ), dtype = np.int16) )

    matrix = pd.DataFrame( np.vstack(matrix), columns = cols )
    matrix["date_block_num"] = matrix["date_block_num"].astype(np.int8)
    matrix["shop_id"] = matrix["shop_id"].astype(np.int8)
    matrix["item_id"] = matrix["item_id"].astype(np.int16)
    matrix.sort_values( cols, inplace = True )

    return matrix


def combineDf(df_a:pd.DataFrame, df_b:pd.DataFrame, cols:list):
    matrix = pd.merge( df_a, df_b, on = cols, how = "left" )
    return matrix


def splitDate(df:pd.DataFrame):
    df["month"] = df["date_block_num"] % 12
    days = pd.Series([31,28,31,30,31,30,31,31,30,31,30,31])
    df["days"] = df["month"].map(days).astype(np.int8)

    return df

#############################################################################################
df_train = pd.read_csv('./data/sales_train.csv')
df_items = pd.read_csv('./data/items.csv')
df_cate = pd.read_csv('./data/item_categories.csv')
df_test = pd.read_csv('./data/test.csv')
df_shop = pd.read_csv('./data/shops.csv')

#############################################################################################
df_train = train_preprocess(df_train)
df_test = test_preprocess(df_test)
df_shop = shop_preprocess(df_shop)
df_cate = category_preprocess(df_cate)
df_items = items_preprocess(df_items)

#############################################################################################
df_total = createTotalDataframe()

df_total = combineDf(df_total, df_shop, ['shop_id'])
df_total = combineDf(df_total, df_items, ['item_id'])
df_total = combineDf(df_total, df_cate, ['item_category_id'])


df_total = splitDate(df_total)