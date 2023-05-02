import json, glob, os, random
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime
from copy import deepcopy
from collections import defaultdict
#from sklearn.metrics import f1_score, accuracy_score
from seqeval.metrics import classification_report
from transformers import AutoTokenizer, AutoModelForMaskedLM,BertTokenizer, BertModel

from BIO_dataset import dataset
from custom_loss import DiceLoss
from utils import *
from logger import *
#from bio_base import *
from aa import *


logger = logging.getLogger(__name__)


#train
def run(args, logger):
    """
        Run BIO tagging task
    """

    dev_res_list = []
    test_res_list = []
    train_res_list = []
    results = {}

    #Init the finalModel
    logger.info("building AA Model: {}".format(args.base_model))
    aa_model = AAModel(args)


    logger.info("loading data")
    df_train, df_val, df_test = aa_model.prepare_data()


    train_dataset = DataSequence(df_train, args.max_length, args.base_model)
    val_dataset = DataSequence(df_val, args.max_legnth, args.base_model)
    test_dataset = DataSequence(df_test, args.max_length, args.base_model)

    train_dataloader = DataLoader(train_dataset, num_workers=4, batch_size=args.train_batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, num_workers=4, batch_size=args.train_batch_size)
    test_dataloader = DataLoader(test_dataset, num_workers=4, batch_size=args.test_batch_size)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    optimizer = torch.optim.Adam(params=aa_model.parameters(), lr=args.learning_rate)

    if use_cuda:
        model = model.cuda()

    best_acc = 0
    best_loss = 1000

    for epoch_num in range(args.num_train_epochs):

        total_acc_train = 0
        total_loss_train = 0

        model.train()

        for train_data, train_label in tqdm(train_dataloader):

            train_label = train_label.to(device)
            mask = train_data['attention_mask'].squeeze(1).to(device)
            input_id = train_data['input_ids'].squeeze(1).to(device)

            optimizer.zero_grad()
            loss, logits = model(input_id, mask, train_label)

            for i in range(logits.shape[0]):

              logits_clean = logits[i][train_label[i] != -100]
              label_clean = train_label[i][train_label[i] != -100]

              predictions = logits_clean.argmax(dim=1)
              acc = (predictions == label_clean).float().mean()
              total_acc_train += acc
              total_loss_train += loss.item()

            loss.backward()
            optimizer.step()

        model.eval()

        total_acc_val = 0
        total_loss_val = 0

        for val_data, val_label in val_dataloader:

            val_label = val_label.to(device)
            mask = val_data['attention_mask'].squeeze(1).to(device)
            input_id = val_data['input_ids'].squeeze(1).to(device)

            loss, logits = model(input_id, mask, val_label)

            for i in range(logits.shape[0]):

              logits_clean = logits[i][val_label[i] != -100]
              label_clean = val_label[i][val_label[i] != -100]

              predictions = logits_clean.argmax(dim=1)
              acc = (predictions == label_clean).float().mean()
              total_acc_val += acc
              total_loss_val += loss.item()

        val_accuracy = total_acc_val / len(df_val)
        val_loss = total_loss_val / len(df_val)

        print(f'Epochs: {epoch_num + 1} | Loss: {total_loss_train / len(df_train): .3f} | Accuracy: {total_acc_train / len(df_train): .3f} | Val_Loss: {total_loss_val / len(df_val): .3f} | Accuracy: {total_acc_val / len(df_val): .3f}')
    aa_model.save('init_supervised')

    aa_model_preds, aa_model_conf = aa_model.predict(test_dataloader)




    # Store the final results
    logger.info("Final Results")

    results['final_res'] = aa_model_preds
    return results



#batch
class Batch():
    def __init__(self, data, idx, batch_size, device):
        cur_batch = data[idx:idx+batch_size]
        src = torch.tensor([x[0] for x in cur_batch])
        seg = torch.tensor([x[1] for x in cur_batch])
        label = torch.tensor([x[2] for x in cur_batch])
        mask_src = 0 + (src != 0)

        self.src = src.to(device)
        self.seg= seg.to(device)
        self.label = label.to(device)
        self.mask_src = mask_src.to(device)

    def get(self):
        return self.src, self.seg, self.label, self.mask_src


#predict
def prediction(dataset, model, args):
    preds = []
    golds = []
    model.eval()
    for i in range(0, len(dataset), args.batch_size):
        src, seg, label, mask_src = Batch(dataset, i, args.batch_size, args.device).get()
        preds += model.predict(src, seg, mask_src)
        golds += label.cpu().data.numpy().tolist()
    return f1_score(golds, preds, average='macro'), preds

def main():
    args_parser = argparse.ArgumentParser()
    # Main Arguments
    parser.add_argument('--dataset', help='Dataset name', type=str, default='Twitter15')
    parser.add_argument("--datapath", help="Path to base dataset folder", type=str, default='../data')
    parser.add_argument("--adapter_path", help="Pretrained adapter path", type=str, default='../adapter')

    #Arguments for experiments
    args_parser.add_argument('--base_model', default='MBERT', choices=['MBERT', 'XLMR','BERT','RobERTa'], help='select one of models')
    args_parser.add_argument('--num_labels', type=int, default=3, help='for appraisal/polarity token classification')
    args_parser.add_argument('--lower_case', default=False, help='whether do lower case when tokenizing')
    args_parser.add_argument('--pretrain_data_path', default='fp_model/',help='path to further pretrained base models')
    args_parser.add_argument('--max_length', type=int, default=128, help='maximum token length for each input sequence')
    args_parser.add_argument('--train_batch_size', type=int, default=20, help='train batch size')
    args_parser.add_argument('--test_batch_size', type=int, default=10, help='test batch size')
    args_parser.add_argument('--learning_rate', type=float, default=1e-5, help='learning rate')
    args_parser.add_argument('--weight_decay', type=int, default=0, help='weight decay')
    args_parser.add_argument('--adam_epsilon', type=float, default=1e-8, help='adam epsilon')
    args_parser.add_argument('--max_grad_norm', type=float, default=10)
    args_parser.add_argument('--num_train_epochs', type=int, default=5, help='total epoch')
    args_parser.add_argument('--warmup_steps', type=int, default=0, help='warmup_steps, the default value is 10% of total steps')
    args_parser.add_argument('--logging_steps', type=int, default=200, help='report stats every certain steps')
    args_parser.add_argument('--seed', type=int, default=42, help='set up a seed for reproductibility')
    args_parser.add_argument('--patience', type=int, default=5, help='patience for early stopping')
    args_parser.add_argument('--gpu', default=True, help='whether to use GPU')
    args_parser.add_argument('--logdir', help='Directory to store logs', type=str, default='logs/')
    args_parser.add_argument('--output_dir',help='Directory to store output results', type=str, default='res/')
    args = args_parser.parse_args()


    #setupCUDA
    if args.gpu:
        device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
        args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Start Experiment
    now = datetime.now()
    date_time = now.strftime("%Y_%m_%d-%H_%M")

    args.experiment_folder = os.path.join(args.experiment_folder, args.dataset)
    args.logdir = os.path.join(args.experiment_folder, args.logdir)
    experiment_dir = str(Path(args.logdir).parent.absolute())


    if os.path.exists(args.logdir):
        shutil.rmtree(args.logdir)

    if args.debug:
        args.logdir = os.path.join(args.experiment_folder, 'debug')
    else:
        args.logdir = args.logdir + "/" + date_time + "_st{}".format(args.student_name.upper())

    os.makedirs(args.logdir, exist_ok=True)
    logger = get_logger(logfile=os.path.join(args.logdir, 'log.log'))

    logger.info("*** EXPERIMENT Start *** with args={}".format(args))
    run(args, logger=logger)
    close(logger)

    os.makedirs(args.output_dir, exist_ok=True)
    all_results = summarize_results(args.output_dir, args.dataset, args.base_model)
    print("*** Results summary (metric={}): {} ***".format(args.metric, all_results))

if __name__ == "__main__":
    main()
