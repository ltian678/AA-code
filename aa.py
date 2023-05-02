from utils import *
#from bio_base import *
from transformers import RobertaConfig, RobertaModel,RobertaTokenizer,XLMRobertaTokenizer,XLMRobertaModel
from transformers import BertConfig, BertTokenizer, BertModel
from transformers import RobertaForTokenClassification,BertForTokenClassification,XLMRobertaForTokenClassification,XLMRobertaConfig


model_dict = { 'MBERT': 'bert-base-multilingual-cased',
               'XLMR': 'xlm-roberta-base',
               'BERT': 'bert-base-cased',
               'RoBERTa': 'roberta-base'
               }

tokenizer_dict = {'BERT':'BertTokenizer',
                  'RoBERTa': 'RobertaTokenizer',
                  'MBERT': 'BertTokenizer',
                  'XLMR': 'XLMRobertaTokenizer'
                }

base_model_dict = { 'MBERT': 'BertForTokenClassification',
                    'XLMR': 'XLMRobertaForTokenClassification',
                    'BERT': 'BertForTokenClassification',
                    'RoberTa': 'RobertaForTokenClassification',
                }

model_config_dict = { 'MBERT': 'BertConfig',
               'XLMR': 'XLMRobertaConfig',
               'BERT': 'BertConfig',
               'RoBERTa': 'RobertaConfig'
               }


def align_label(texts, labels, max_len):
    tokenized_inputs = tokenizer(texts, padding='max_length', max_length=max_len, truncation=True)

    word_ids = tokenized_inputs.word_ids()

    previous_word_idx = None
    label_ids = []

    for word_idx in word_ids:

        if word_idx is None:
            label_ids.append(-100)

        elif word_idx != previous_word_idx:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]])
            except:
                label_ids.append(-100)
        else:
            try:
                label_ids.append(labels_to_ids[labels[word_idx]] if label_all_tokens else -100)
            except:
                label_ids.append(-100)
        previous_word_idx = word_idx

    return label_ids

class DataSequence(torch.utils.data.Dataset):

    def __init__(self, df, max_len, base_model):
        tokenizer = tokenizer_dict[base_model].from_pretrained(model_dict[base_model])
        lb = [i.split(" ") for i in df['label'].values.tolist()]
        txt = df['text'].values.tolist()
        self.texts = [tokenizer(str(i),
                               padding='max_length', max_length = max_len, truncation=True, return_tensors="pt") for i in txt]
        self.labels = [align_label(i,j) for i,j in zip(txt, lb)]

    def __len__(self):

        return len(self.labels)

    def get_batch_data(self, idx):

        return self.texts[idx]

    def get_batch_labels(self, idx):

        return torch.LongTensor(self.labels[idx])

    def __getitem__(self, idx):

        batch_data = self.get_batch_data(idx)
        batch_labels = self.get_batch_labels(idx)

        return batch_data, batch_labels




class AAModel(torch.nn.Module):
    def __init__(self,args):
        super(AAModel, self).__init__()
        """
        Token classifier with transformer-based LMs
        base_model: define the base multilingual/monolingual transformer encoder
        model_card: encoder version
        do_lower_case: whether do lower case
        """
        self.args = args
        self.gpu = args.gpu
        self.input_data = args.input_data_file
        self.adapter = args.adapter
        self.adapter_path = args.adapter_path
        self.base_model = base_model_dict[args.bert_model]
        self.base_model_config = model_config_dict[args.bert_model]
        self.bert = self.base_model.from_pretrained(model_dict[args.bert_model],num_labels = args.num_labels,output_attentions = False,return_dict=False)
        self.base_tokenizer = tokenizer_dict[args.bert_model]
        self.lr = args.learning_rate
        self.do_lower_case = args.lower_case
        self.num_labels = args.num_labels
        self.max_length = args.max_length
        self.batch_size = args.batch_size
        self.num_train_epochs = args.num_train_epochs
        self.warmup_steps = args.warmup_steps
        self.train_batch_size = args.train_batch_size
        self.test_batch_size = args.test_batch_size
        self.optimizer = None
        self.scheduler = None
        self.linear_app = nn.Linear(self.bert.config.hidden_size, args.num_labels)
        self.linear_pol = nn.Linear(self.bert.config.hidden_size, args.num_labels)
        self.dropout = nn.Dropout(0.2)
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=args.num_labels, reduction='sum')
        self.output = '../saved_models/'


    def __init__(self):

        super(AAModel, self).__init__()

    @property
    def device(self):
        if self.gpu:
            return torch.device('cuda')
        else:
            return torch.device('cpu')

    def prepare_data(self):
        df = gen_df(self.input_data)
        label_all_tokens = False

        labels = [i.split() for i in df['label'].values.tolist()]
        unique_labels = set()

        for lb in labels:
                [unique_labels.add(i) for i in lb if i not in unique_labels]
        labels_to_ids = {k: v for v, k in enumerate(unique_labels)}
        ids_to_labels = {v: k for v, k in enumerate(unique_labels)}

        df_train, df_val, df_test = np.split(df.sample(frac=1, random_state=42),
                                    [int(.8 * len(df)), int(.9 * len(df))])
        return df_train, df_val, df_test

    def align_label(self, texts, labels):
        tokenizer = self.base_tokenizer
        tokenized_inputs = tokenizer(texts, padding='max_length', max_length=self.max_length, truncation=True)

        word_ids = tokenized_inputs.word_ids()

        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:

            if word_idx is None:
                label_ids.append(-100)

            elif word_idx != previous_word_idx:
                try:
                    label_ids.append(labels_to_ids[labels[word_idx]])
                except:
                    label_ids.append(-100)
            else:
                try:
                    label_ids.append(labels_to_ids[labels[word_idx]] if label_all_tokens else -100)
                except:
                    label_ids.append(-100)
            previous_word_idx = word_idx

        return label_ids


    def forward(self, input_id, mask, label):

        outputs = self.bert(input_ids=input_id, attention_mask=mask, labels=label, return_dict=False)
        x1 = self.linear1(output)
        x2 = self.linear2(output)
        x1 = F.log_softmax(x1, self.num_labels)
        x2 = F.log_softmax(x2, self.num_labels)
        return x1, x2


    def predict(self, dataset, student_features=None):
        pred_labels, pred_conf = self.apply(dataset)
        return pred_labels, pred_conf

    def save(self, name):
        file = "{}/{}.pkl".format(self.output, name)
        if not os.path.exists(self.output):
            os.makedirs(self.output)
        logging.info("Saving model to {}".format(name))
        state = {
            "model": self.bert.state_dict()
        }
        torch.save(state, file)
