from utils import *
from transformers import RobertaTokenizer,TrainingArguments, AdapterTrainer, EvalPrediction
import numpy as np


MAX_LEN = args.max_length

#Adapter Pretraining
def encode_batch(batch):
  """Encodes a batch of input data using the model tokenizer."""
  return tokenizer(batch["clean_tweets_string"], max_length=MAX_LEN, truncation=True, padding="max_length")


def compute_accuracy(p: EvalPrediction):
  preds = np.argmax(p.predictions, axis=1)
  return {"acc": (preds == p.label_ids).mean()}



def pretrain_adapter(args):
    tokenizer = find_tokenizer(args.base_model)
    ds = read_data_from(args.input_data_file)
    #Encode the input data
    dataset = ds.map(encode_batch, batched=True)
    # The transformer expect the target class column to be named "labels"
    dataset.rename_column_("encoded_label","labels")
    #Transform to pytorch tensor and only output the required columns
    dataset.set_format(type='torch',columns=['input_ids','attention_mask','labels'])

    config, model = find_model_config(base_model)

    #init the new adapter
    model.add_adapter(args.adapter_name)
    # add a matching classification head
    model.add_classification_head(args.adapter_name, num_labels=2,id2label={0:"positive",1:"negative"})
    #activate the adapter
    model.train_adapter(args.adapter_name)

    training_args = TrainingArguments(
        learning_rate=1e-4,
        num_train_epochs=6,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        logging_steps=200,
        output_dir="./training_output",
        overwrite_output_dir=True,
        # The next line is important to ensure the dataset labels are properly passed to the model
        remove_unused_columns=False)

    #init the Trainer
    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        compute_metrics=compute_accuracy)

    trainer.train()

    eval_trainer = AdapterTrainer(
        model=model,
        args=TrainingArguments(output_dir="./test_output", remove_unused_columns=False,),
        eval_dataset=dataset["test"],
        compute_metrics=compute_accuracy)

    eval_trainer.evaluate()
