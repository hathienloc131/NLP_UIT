

import pandas as pd

def get_analysis(data_path="data/ise-dsc01-train.json"):
    df = pd.read_json(data_path, orient='index')
    count_label = df.verdict.value_counts()
    count_label =  count_label.min() / count_label
    print(count_label)
    return count_label.tolist()

# print(df.loc[46648].claim)

