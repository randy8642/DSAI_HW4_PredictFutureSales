import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('./data/sales_train.csv')
print(df)
# df['date'] = [datetime.strptime(date_str, "%d.%m.%Y") for date_str in df['date'].values]

# plt.plot(df['date'], df['item_cnt_day'], 'x')
# plt.show()


d = np.zeros([33,1])

for i in df.index:
    row = df.iloc[i]
    block_num = row['date_block_num']
    total_price = row['item_price'] * row['item_cnt_day']

    d[block_num,0] += total_price

plt.plot(d.flatten())
plt.show()



