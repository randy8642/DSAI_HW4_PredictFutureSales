# DSAI_HW4_PredictFutureSales
NCKU DSAI course homework

## 前置工作
### 作業說明
* 說明連結與kaggle資料集
[https://www.kaggle.com/c/competitive-data-science-predict-future-sales/overview](https://www.kaggle.com/c/competitive-data-science-predict-future-sales/overview)

* 目標\
以過去34個月各商家的不同商品之交易量，\
預測未來一個月特定商家的數種商品之月銷量。

### 環境
* python 3.8.5
* Win 10

### 使用方式
1. 進入專案資料夾\
`cd /d [path/to/this/project]` 

2. 安裝所需套件\
`pip install -r requirements.txt`

3. 進行資料前處理，得到處理後的資料集 (`./data/Inputs.npz`)\
   **(亦可直接進行第4步驟)**\
`python Data_preprocess.py`   

4. 訓練模型，得到訓練好的模型 (`XGmodel`)\
`python main_train.py`

5. 執行預訓練的模型，並得到預測資料 (`submission.csv`)\
`python main.py`

### 最終分數 (RMSE)
![Imgur](https://i.imgur.com/dFeuRl4.png)


### 期末報告
![](https://github.com/randy8642/DSAI_HW4_PredictFutureSales/blob/main/img/001.jpg)
![](https://github.com/randy8642/DSAI_HW4_PredictFutureSales/blob/main/img/002.jpg)
![](https://github.com/randy8642/DSAI_HW4_PredictFutureSales/blob/main/img/003.jpg)
![](https://github.com/randy8642/DSAI_HW4_PredictFutureSales/blob/main/img/004.jpg)
![](https://github.com/randy8642/DSAI_HW4_PredictFutureSales/blob/main/img/005.jpg)
![](https://github.com/randy8642/DSAI_HW4_PredictFutureSales/blob/main/img/006.jpg)
![](https://github.com/randy8642/DSAI_HW4_PredictFutureSales/blob/main/img/007.jpg)
![](https://github.com/randy8642/DSAI_HW4_PredictFutureSales/blob/main/img/008.jpg)