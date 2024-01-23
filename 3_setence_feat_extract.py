import os
import numpy as np

from path_datasets_and_weights import path_internal_dataset
import pandas as pd
from dataset.constants import REPORT_KEYS, CLIN_KEYS, REPORT_KEYS_raw
from models.text_encoder import text_feat_encoder

def testing(
    text_feat_encoder,
    datasets_df,
):
    text_feats_all = []
    for i, data in enumerate(datasets_df.iterrows()):
        print(i)
        index = data[0]
        data_ = data[1]
        sentences_list = [data_[i] for i in REPORT_KEYS]
        report_feat = text_feat_encoder.forward(sentences_list)
        text_feats_all.append(report_feat[None])
    text_feats_all = np.concatenate(text_feats_all)
    return text_feats_all


def main():
    usecols = ["days", "death", ]
    usecols += CLIN_KEYS
    usecols += REPORT_KEYS
    usecols += REPORT_KEYS_raw
    encoder = text_feat_encoder()
    for setname in ['brown',]:
    # setname = 'brown'
        datasets_df = pd.read_excel(os.path.join(path_internal_dataset, setname+'_table_w_report_split_corr.xlsx'), usecols=usecols)
        text_feats_all = testing(
            text_feat_encoder=encoder,
            datasets_df=datasets_df,
        )

        np.save(setname+'_5text_feats_gatortron_medium.npy', text_feats_all.astype(np.float32))

if __name__ == "__main__":
    main()
