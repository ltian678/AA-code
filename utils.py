import datasets
from datasets import Dataset, DatasetDict
import pandas as pd
from transformers import RobertaTokenizer, BertTokenizer
from transformers import RobertaConfig, RobertaModelWithHeads
from transformers import AutoModelForSequenceClassification
from transformers import RobertaForTokenClassification


def gen_df(input_pkl_file):
    df = pd.read_pickle(input_pkl_file)
    return df

def read_data_from(input_pkl_file):
    df = pd.read_pickle(input_pkl_file)
    train_df = df[df['train_test_vali']=='train']
    vali_df = df[df['train_test_vali']=='vali']
    test_df = df[df['train_test_vali']=='test']

    train_dataset = Dataset.from_pandas(train_df)
    vali_dataset = Dataset.from_pandas(vali_df)
    test_dataset = Dataset.from_pandas(test_df)

    ds = DatasetDict()

    ds['train'] = train_dataset
    ds['validation'] = vali_dataset
    ds['test'] = test_dataset

    return ds


def find_tokenizer(base_model):
    if base_model == 'BERT':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    if base_model == 'RoBERTa':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    return tokenizer


def find_model_config(base_model):
    if base_model == 'BERT':
        model, config
    if base_model == 'RoBERTa':
        config = RobertaConfig.from_pretrained("roberta-base",num_labels=2)
        model = RobertaModelWithHeads.from_pretrained("roberta-base",config=config)
    return config, model


def find_cls_model(base_model):
    if base_model == 'BERT':
        cls_model = BertForTokenClassification
    if base_model == 'RoBERTa':
        cls_model = RobertaForTokenClassification
    return cls_model
