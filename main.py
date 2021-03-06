import pandas as pd
import pickle
import numpy as np
import time

def main():
    tStart = time.time()
    x = np.load('./data/Inputs.npz')
    # def _inputs(x):
    X_test = (x['X_test'])
    model_loaded = pickle.load(open('XGmodel', "rb"))

    pred_tes = model_loaded.predict(X_test)
    id_list = np.arange(0, len(pred_tes), 1).astype(str)
    D = np.vstack([id_list, pred_tes]).T
    df = pd.DataFrame(D, columns=["ID", "item_cnt_month"])
    df.to_csv('submission.csv', index=False)

if __name__=='__main__':
    main()