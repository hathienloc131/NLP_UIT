from underthesea import sent_tokenize
import pandas as pd
import re
from random import shuffle
CLASSES = {"NEI":1, "SUPPORTED":0, "REFUTED":2}


def preprocess_text(text: str) -> str:    
    text = re.sub(r"['\",\.\?:\-!]", "", text)
    text = text.strip()
    text = " ".join(text.split())
    text = text.lower()
    return text

def load_datasets(filename, is_test=False):
    raw_data = pd.read_json(filename, orient='index')
    print(raw_data.index)
    datasets = []
    max_sent_len = 0
    c = 0
    
    for e, row in enumerate(raw_data.itertuples()):
        data = row._asdict()
        # sent_tmp = sent_tokenize(data["context"])
        sent_tmp = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.)(\s|[A-Z].*)', data["context"])
        for sent in sent_tmp:
            if len(sent) <= 1:
                sent_tmp.remove(sent)
        # idx_wrong = [i for i in range(len(sent_tmp)) if "!" in sent_tmp[i] or "?" in sent_tmp[i]]
        # len_sent = len(sent_tmp)
        # for idx in reversed(idx_wrong):
        #     if idx == len_sent - 1:
        #         continue
        #     sent_tmp[idx] = sent_tmp[idx] + sent_tmp[idx + 1]
        #     # print(sent_tmp[idx])
        #     sent_tmp.pop(idx + 1)

        max_sent_len = max(max_sent_len, len(sent_tmp))
        if (e > 15000):
            print(raw_data.index[e])
            break
        if not is_test:
            data["verdict"] = CLASSES[data["verdict"]]
            data["evidence_index"] = 0
            if data["evidence"] is None:
                data["evidence"] = ""
                c+=1
            else:
                pre_c = c
                for i, sentence in enumerate(sent_tmp):
                    if (preprocess_text(data["evidence"]) == preprocess_text(sentence)):
                        data["evidence_index"] = i + 1
                        c+=1
        datasets.append(data)
    # shuffle(datasets)
    print(c)
    print(len(datasets))
    return datasets, max_sent_len
    