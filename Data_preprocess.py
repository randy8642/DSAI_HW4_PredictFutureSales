import time
import pandas as pd
import numpy as np
from itertools import product
import copy
import re
#LabelEncoder().fit_transform([cate])

def train_preprocess(df: pd.DataFrame):
    # 移除離群值
    df = df[(df['item_price'] < 300000) & (df['item_cnt_day'] < 1000)]
    df = df[df['item_price'] > 0].reset_index(drop=True)
    df.loc[df['item_cnt_day'] < 1, 'item_cnt_day'] = 0

    # 合併相同商店(名稱相同)
    same_shopId = [[0, 57], [1, 58], [10, 11]]
    for pair in same_shopId:
        df.loc[df['shop_id'] == pair[0], 'shop_id'] = pair[1]

    # 加上銷售額
    df['revenue'] = df['item_price'] * df['item_cnt_day']

    return df


def test_preprocess(df: pd.DataFrame):
    # 合併相同商店(名稱相同)
    same_shopId = [[0, 57], [1, 58], [10, 11]]
    for pair in same_shopId:
        df.loc[df['shop_id'] == pair[0], 'shop_id'] = pair[1]

    return df


def shop_preprocess(df: pd.DataFrame):

    # 切割商店名稱 => 所在地 & 類型
    df['city'] = df['shop_name'].str.split(' ').map(lambda x: ' '.join(x[0:-2]))
    df['category'] = df['shop_name'].str.split(' ').map(lambda x: x[-2])

    df['city'] = df['city'].apply(lambda x: re.sub('[!]', '', x))

    # 合併過小的類別
    category = []
    for cat in df['category'].unique():
        if len(df[df['category'] == cat]) >= 5:
            category.append(cat)
    df['category'] = df['category'].apply(
        lambda x: x if (x in category) else 'other')

    # 標籤(文字) => 索引(類別編號)
    _, df['shopcategory_id'] = np.unique(df['category'], return_inverse=True)
    _, df['shop_city_id'] = np.unique(df['city'], return_inverse=True)

    # df['shopcategory_id'] = LabelEncoder().fit_transform(df['category'])
    # df['shop_city_id'] = LabelEncoder().fit_transform(df['city'])

    # 新df
    df = df[["shop_id", "shopcategory_id", "shop_city_id"]]

    return df


def category_preprocess(df: pd.DataFrame):

    # 切割類別名稱 => 主類別 & 子類別
    df['cate_type'] = df['item_category_name'].apply(
        lambda x: re.split(' ', x)[0]).astype(str)
    df['cate_subtype'] = df['item_category_name'].apply(
        lambda x: re.split(' - ', x)[-1]).astype(str)

    #
    df.loc[ (df['cate_type'] == 'Игровые')| (df['cate_type'] == 'Аксессуары'), 'cate_type' ] = 'Игры'

    # 合併過小的類別
    category = []
    for cat in df['cate_type'].unique():
        if len(df[df['cate_type'] == cat]) >= 5:
            category.append(cat)

    df['cate_type'] = df['cate_type'].apply(
        lambda x: x if (x in category) else 'etc')

    # 標籤(文字) => 索引(類別編號)
    _, df['cate_type_id'] = np.unique(df['cate_type'], return_inverse=True)
    _, df['cate_subtype_id'] = np.unique(df['cate_subtype'], return_inverse=True)

    # df['cate_type_id'] = LabelEncoder().fit_transform(df['cate_type'])
    # df['cate_subtype_id'] = LabelEncoder().fit_transform(df['cate_subtype'])

    # 新df
    df = df[['item_category_id', 'cate_type_id', 'cate_subtype_id']]

    return df


def items_preprocess(df: pd.DataFrame):

    # 切割商品名稱 => 類別1 & 類別2
    def splitNameType1(x):
        results = re.findall(r'\(.*', x)

        if len(results) > 0:
            result = re.sub('[^A-Za-z0-9А-Яа-я]+', ' ', results[0][1:]).lower()
        else:
            result = np.nan

        return result

    def splitNameType2(x):
        results = re.findall(r'\[.*', x)

        if len(results) > 0:
            result = re.sub('[^A-Za-z0-9А-Яа-я]+', ' ', results[0][1:]).lower()
        else:
            result = np.nan

        return result

    def name_correction(x: str):
        x = x.lower()
        x = re.split(r'[\(\[]', x)[0]
        x = re.sub('[^A-Za-z0-9А-Яа-я]+', ' ', x)
        x = re.sub('  ', ' ', x)
        x = x.strip()

        return x

    df['item_type_1'] = df['item_name'].apply(lambda x: splitNameType1(x))
    df['item_type_2'] = df['item_name'].apply(lambda x: splitNameType2(x))

    df.fillna('0', inplace=True)

    # 修正商品名稱
    df["item_name"] = df["item_name"].apply(lambda x: name_correction(x))

    # 切割類別
    df["type"] = df['item_type_2'].apply(lambda x: x[0:8] if x.split(" ")[
                                         0] == "xbox" else x.split(" ")[0])

    # 整理
    df.loc[(df['type'] == 'x360') | (df['type'] == 'xbox360')
           | (df['type'] == 'xbox 360'), 'type'] = 'xbox 360'
    df.loc[df['type'] == '', 'type'] = 'mac'
    df['type'] = df['type'].apply(lambda x: x.replace(' ', ''))
    df.loc[(df['type'] == 'pc') | (df['type'] == 'pс')
           | (df['type'] == 'pc'), 'type'] = 'pc'
    df.loc[df['type'] == 'рs3', 'type'] = 'ps3'

    # 整合較小的類別
    group_sum = df.groupby(['type']).agg({'item_id': 'count'})
    group_sum = group_sum.reset_index()
    drop_cols = []
    for cat in group_sum['type'].unique():
        if group_sum.loc[(group_sum['type'] == cat), 'item_id'].values[0] < 40:
            drop_cols.append(cat)
    df['item_type_2'] = df['item_type_2'].apply(
        lambda x: 'other' if (x in drop_cols) else x)
    df = df.drop(['type'], axis=1)

    # 標籤(文字) => 索引(類別編號)
    _, df['item_type_1_id'] = np.unique(df['item_type_1'], return_inverse=True)
    _, df['item_type_2_id'] = np.unique(df['item_type_2'], return_inverse=True)

    # df['item_type_1_id'] = LabelEncoder().fit_transform(df['item_type_1'])
    # df['item_type_2_id'] = LabelEncoder().fit_transform(df['item_type_2'])

    # 新df
    df = df[['item_id', 'item_category_id', 'item_type_1_id', 'item_type_2_id']]

    return df


def save_dataframe(df: pd.DataFrame):
    print(df)
    df.to_csv('./data/preprocessData.csv', index=False, encoding='utf-8')
    data = df.copy()
    X_train = data[data.date_block_num < 33].drop(['item_cnt_month'], axis=1)
    Y_train = data[data.date_block_num < 33]['item_cnt_month']
    X_valid = data[data.date_block_num == 33].drop(['item_cnt_month'], axis=1)
    Y_valid = data[data.date_block_num == 33]['item_cnt_month']
    X_test = data[data.date_block_num == 34].drop(['item_cnt_month'], axis=1)
    Y_train = Y_train.clip(0, 20)
    Y_valid = Y_valid.clip(0, 20)

    sP = './data/Inputs.npz'
    np.savez_compressed(sP, X_train=X_train, Y_train=Y_train,
                        X_valid=X_valid, Y_valid=Y_valid, X_test=X_test)


def createTotalDataframe(df_src:pd.DataFrame):
    

    matrix = []
    cols = ["date_block_num", "shop_id", "item_id"]
    for i in range(34):
        sales = df_src[df_src['date_block_num'] == i]
        matrix.append(np.array(list(product([i], sales['shop_id'].unique(), sales['item_id'].unique())), dtype=np.int))

    matrix = pd.DataFrame(np.vstack(matrix), columns=cols)
    matrix["date_block_num"] = matrix["date_block_num"].astype(np.int8)
    matrix["shop_id"] = matrix["shop_id"].astype(np.int8)
    matrix["item_id"] = matrix["item_id"].astype(np.int16)
    matrix.sort_values(cols, inplace=True)

    return matrix


def combineDf(df_a: pd.DataFrame, df_b: pd.DataFrame, cols: list):
    matrix = pd.merge(df_a, df_b, on=cols, how="left")
    return matrix


def splitDate(df: pd.DataFrame):
    df["month"] = df["date_block_num"] % 12
    days = pd.Series([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
    df["days"] = df["month"].map(days).astype(np.int8)

    return df


def addMonthCnt(df: pd.DataFrame, df_src: pd.DataFrame):
    group = df_src.groupby(["date_block_num", "shop_id", "item_id"]).agg(
        {"item_cnt_day": ["sum"]})
    group.columns = ["item_cnt_month"]
    group.reset_index(inplace=True)

    cols = ["date_block_num", "shop_id", "item_id"]
    df = pd.merge(df, group, on=cols, how="left")
    df["item_cnt_month"] = df["item_cnt_month"].fillna(0).astype(np.float16)

    return df


def addMonthAvgCnt(df: pd.DataFrame):
    group = df.groupby(["date_block_num"]).agg({"item_cnt_month": ["mean"]})
    group.columns = ['date_avg_item_cnt']
    group.reset_index(inplace=True)

    df = pd.merge(df, group, on=["date_block_num"], how="left")

    return df


def addMonthItemAvgCnt(df: pd.DataFrame):
    group = df.groupby(['date_block_num', 'item_id']).agg(
        {'item_cnt_month': ['mean']})
    group.columns = ['date_item_avg_item_cnt']
    group.reset_index(inplace=True)

    df = pd.merge(df, group, on=['date_block_num', 'item_id'], how='left')

    return df


def addMonthShopsubTypeAvgCnt(df: pd.DataFrame):
    group = df.groupby(['date_block_num', 'shop_id', 'cate_subtype_id']).agg(
        {'item_cnt_month': ['mean']})
    group.columns = ['date_shop_subtype_avg_item_cnt']
    group.reset_index(inplace=True)

    df = pd.merge(df, group, on=['date_block_num',
                  'shop_id', 'cate_subtype_id'], how='left')

    return df


def addMonthCityAvgCnt(df: pd.DataFrame):
    group = df.groupby(['date_block_num', 'shop_city_id']
                       ).agg({'item_cnt_month': ['mean']})
    group.columns = ['date_city_avg_item_cnt']
    group.reset_index(inplace=True)

    df = pd.merge(df, group, on=['date_block_num', "shop_city_id"], how='left')

    return df


def addMonthCityItemAvgCnt(df: pd.DataFrame):
    group = df.groupby(['date_block_num', 'item_id', 'shop_city_id']).agg(
        {'item_cnt_month': ['mean']})
    group.columns = ['date_item_city_avg_item_cnt']
    group.reset_index(inplace=True)

    df = pd.merge(df, group, on=['date_block_num',
                  'item_id', 'shop_city_id'], how='left')

    return df


def revenue(df: pd.DataFrame, df_src: pd.DataFrame):
    #
    group = df_src.groupby(['date_block_num', 'shop_id']
                           ).agg({'revenue': ['sum']})
    group.columns = ['date_shop_revenue']
    group.reset_index(inplace=True)

    df = pd.merge(df, group, on=["date_block_num", "shop_id"], how="left")
    df['date_shop_revenue'] = df['date_shop_revenue'].astype(np.float32)

    #
    group = group.groupby(["shop_id"]).agg({"date_block_num": ["mean"]})
    group.columns = ["shop_avg_revenue"]
    group.reset_index(inplace=True)

    df = pd.merge(df, group, on=['shop_id'], how="left")

    #
    df['shop_avg_revenue'] = df['shop_avg_revenue'].astype(np.float32)
    df['delta_revenue'] = (df['date_shop_revenue'] -
                           df['shop_avg_revenue']) / df['shop_avg_revenue']
    df['delta_revenue'] = df['delta_revenue'].astype(np.float32)

    #
    df = addLag(df, 'delta_revenue', [1])
    df['delta_revenue_lag_1'] = df['delta_revenue_lag_1'].astype(np.float32)
    df.drop(['date_shop_revenue', 'shop_avg_revenue',
            'delta_revenue'], axis=1, inplace=True)

    return df


def addLag(df: pd.DataFrame, col: str, lags: list):

    assert sum([type(x) == int for x in lags]) == len(
        lags), '非所有lags為type(int)'

    tmp = df[["date_block_num", "shop_id", "item_id", col]]

    for i in lags:
        shifted = copy.deepcopy(tmp)
        shifted.columns = ["date_block_num",
                           "shop_id", "item_id", col + "_lag_"+str(i)]
        shifted['date_block_num'] = shifted['date_block_num'] + i
        df = pd.merge(df, shifted, on=[
                      'date_block_num', 'shop_id', 'item_id'], how='left')

    return df


def createTest(df: pd.DataFrame, df_src: pd.DataFrame):
    cols = ["date_block_num", "shop_id", "item_id"]

    df_src['date_block_num'] = 34
    df = pd.concat([df, df_src.drop(["ID"], axis=1)],  ignore_index=True, sort=False, keys=cols)
    df['shop_id'] = df['shop_id'].astype(np.int8)
    df['item_id'] = df['item_id'].astype(np.int16)
    df['date_block_num'] = df['date_block_num'].astype(np.int8)
    df.fillna(0, inplace=True)

    return df


def addFirstSale(df:pd.DataFrame):
    df["item_shop_first_sale"] = df["date_block_num"] - df.groupby(["item_id","shop_id"])["date_block_num"].transform('min')
    df["item_first_sale"] = df["date_block_num"] - df.groupby(["item_id"])["date_block_num"].transform('min')

    return df


def del_nan_Lag(df:pd.DataFrame):
    df = df[df["date_block_num"] > 3]

    return df

#############################################################################################
def main():

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
    df_total = createTotalDataframe(df_train)

    df_total = createTest(df_total, df_test)

    # 合併
    df_total = combineDf(df_total, df_shop, ['shop_id'])
    df_total = combineDf(df_total, df_items, ['item_id'])
    df_total = combineDf(df_total, df_cate, ['item_category_id'])
    df_total['shop_city_id'] = df_total['shop_city_id'].astype(np.int8)
    df_total['shopcategory_id'] = df_total['shopcategory_id'].astype(np.int8)
    df_total['item_category_id'] = df_total['item_category_id'].astype(np.int8)
    df_total['cate_subtype_id'] = df_total['cate_subtype_id'].astype(np.int8)
    df_total['item_type_2_id'] = df_total['item_type_2_id'].astype(np.int8)
    df_total['item_type_1_id'] = df_total['item_type_1_id'].astype(np.int16)
    df_total['cate_type_id'] = df_total['cate_type_id'].astype(np.int8)


    # 增加銷售量feature
    df_total = addMonthCnt(df_total, df_train)
    df_total = addMonthAvgCnt(df_total)
    df_total['date_avg_item_cnt'] = df_total['date_avg_item_cnt'].astype(np.float16)
    df_total = addMonthItemAvgCnt(df_total)
    df_total['date_item_avg_item_cnt'] = df_total['date_item_avg_item_cnt'].astype(np.float16)
    df_total = addMonthShopsubTypeAvgCnt(df_total)
    df_total['date_shop_subtype_avg_item_cnt'] = df_total['date_shop_subtype_avg_item_cnt'].astype(np.float16)
    df_total = addMonthCityAvgCnt(df_total)
    df_total['date_city_avg_item_cnt'] = df_total['date_city_avg_item_cnt'].astype(np.float16)
    df_total = addMonthCityItemAvgCnt(df_total)
    df_total['date_item_city_avg_item_cnt'] = df_total['date_item_city_avg_item_cnt'].astype(np.float16)

    df_total = addLag(df_total, 'item_cnt_month', [1, 2, 3])
    df_total = addLag(df_total, 'date_avg_item_cnt', [1])
    df_total = addLag(df_total, 'date_item_avg_item_cnt', [1, 2, 3])
    df_total = addLag(df_total, 'date_shop_subtype_avg_item_cnt', [1])
    df_total = addLag(df_total, 'date_city_avg_item_cnt', [1])
    df_total = addLag(df_total, 'date_item_city_avg_item_cnt', [1])

    df_total.drop(['date_avg_item_cnt', 'date_item_avg_item_cnt', 'date_shop_subtype_avg_item_cnt', 'date_city_avg_item_cnt',
                'date_item_city_avg_item_cnt'], axis=1, inplace=True)

    # 增加銷售額features
    df_total = revenue(df_total, df_train)

    # 拆解月份
    df_total = splitDate(df_total)

    #
    df_total = addFirstSale(df_total)

    # 刪除前幾筆無法產生Lag feature的項目
    df_total = del_nan_Lag(df_total)

    # NAN -> 0 
    df_total.fillna(0, inplace=True)


    # 儲存
    save_dataframe(df_total)


if __name__ == '__main__':
    tStart = time.time()
    main()
    tEnd = time.time()
    print ("\n" + "It cost {:.4f} sec" .format(tEnd-tStart))