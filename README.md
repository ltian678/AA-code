# Adaptive Adpater Appraisal

## Paper
Task and Sentiment Adaptation for Appraisal Tagging

##  Data and Resource
We use SOCC and POST datasets for rumour detection.

In this repository, we links to the public available dataset SOCC. Please `git clone`
In this repository, we do not provide you with the raw input data. Please download the datasets from the following links.

| Dataset | Link |
| --- | --- |
| SOCC | https://github.com/sfu-discourse-lab/SOCC ) |

## Dependencies
1. Python 3.6
2. Run `pip install -r requirements.txt`

## Pre-training LM script:

Clone the transformer to local directory
```
git clone https://github.com/huggingface/transformers.git
```


For further pre-training the language models:
```
python transformers/examples/language-modeling/run_language_modeling.py ,
        --output_dir='ML_MBERT_TAPT',
        --model_type=bert ,
        --model_name_or_path=bert-base-multilingual-cased,
        --do_train,
        --overwrite_output_dir,
        --train_data_file='train.txt',
        --do_eval,
        --block_size=256,
        --eval_data_file='vali.txt',
        --mlm"
```




#If you find this code useful, please let us know and cite our paper.
