

import pandas as pd
import sys
import os
if __name__ == '__main__':
    data_path = sys.argv[1]
    folder_path = sys.argv[2]
    df = pd.read_json(data_path, orient='index')
    df = df.sample(frac=1)
    num_row = len(df.index)
    num_train = int(0.9 * num_row)
    train_df = df.iloc[: num_train]
    dev_df = df.iloc[num_train : ]


    train_df.to_json(os.path.join(folder_path, "train.json"),orient="index", force_ascii=False)
    dev_df.to_json(os.path.join(folder_path, "dev.json"),orient="index", force_ascii=False)

    