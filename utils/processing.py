from underthesea import sent_tokenize
import pandas as pd
import re
from random import shuffle
CLASSES = {"NEI":1, "SUPPORTED":0, "REFUTED":2}

def load_datasets(filename, is_test=False):
    raw_data = pd.read_json(filename, orient='index')
    datasets = []
    max_sent_len = 0
    
    for row in raw_data.itertuples():
        data = row._asdict()
        data["context"] = sent_tokenize(data["context"])
        max_sent_len = max(max_sent_len, len(data["context"]))
        if not is_test:
            data["verdict"] = CLASSES[data["verdict"]]
            data["evidence_index"] = 0
            if data["evidence"] is None:
                data["evidence"] = ""
            else:
                for i, sentence in enumerate(data["context"]):
                    if (data["evidence"] in sentence):
                        data["evidence_index"] = i + 1
        datasets.append(data)
    shuffle(datasets)

    return datasets, max_sent_len
    