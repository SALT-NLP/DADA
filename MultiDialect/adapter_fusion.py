import os
import json
import torch
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import load_dataset
from collections import defaultdict
from transformers.adapters.composition import Fuse
import transformers
from transformers import RobertaTokenizer, RobertaConfig, RobertaAdapterModel, TrainingArguments, AdapterTrainer, EvalPrediction, set_seed
from huggingface_hub import login
from torch.utils.data import DataLoader
from scipy.special import softmax

set_seed(3407)
transformers.logging.set_verbosity_error()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
login('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')

task2id2label={
    'sst2': { 0: "👎", 1: "👍"},
    'mnli' :{0: 'entailment', 1: 'neutral', 2: 'contradiction'},
}

def compute_accuracy(p: EvalPrediction):
    preds = np.argmax(p.predictions, axis=1)
    return {"acc": (preds == p.label_ids).mean()}

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', default=None)
    parser.add_argument('--task_name', default='MNLI')
    parser.add_argument('--adapters_path', default='./Adapters')
    parser.add_argument('--transformation_rules_path', default='transformation_rules_list.json')
    parser.add_argument('--num_transformation_rules', type=int, default=None)
    parser.add_argument('--job_id', type=int, default='2')
    parser.add_argument("--learning_rate", type=float, default=2.5e-5)
    parser.add_argument("--train_epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()
    print(' '.join(f'{k}={v}' for k, v in vars(args).items()))


    # setting
    task_name = args.task_name
    task_name = task_name.lower().replace('-','')
    transformation_rules_path = args.transformation_rules_path
    num_transformation_rules = args.num_transformation_rules
    learning_rate = args.learning_rate
    train_epochs = args.train_epochs
    batch_size = args.batch_size

    adapter_names = json.load(open(transformation_rules_path, 'r'))
    if num_transformation_rules:
        adapter_names = adapter_names[:num_transformation_rules]
    adapter_names = [task_name + "_" + section for section in adapter_names]

    if args.output_path is None:
        output_path = './outputs'
    else: output_path = args.output_path
    adapters_path = args.adapters_path

    train_dataset = 'MULTI'
    test_dialects = ['MULTI', 'AppE', 'ChcE', 'CollSgE', 'IndE', 'VALUE', 'GLUE']
    predict_dialects = ['MULTI', 'AppE', 'ChcE', 'CollSgE', 'IndE', 'VALUE', 'GLUE']
    analysis_split = 'test_matched' if task_name =='mnli' else 'test'
    analysis_split = 'train'

    # data preprocessing
    dataset = load_dataset('liuyanchen1015/' + task_name + '_' + train_dataset)

    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    if task_name == 'mnli':
        def encode_batch(batch):
            """Encodes a batch of input data using the model tokenizer."""
            return tokenizer(batch["premise"], batch["hypothesis"], max_length=128, truncation=True, padding="max_length")
    else:
        def encode_batch(batch):
            """Encodes a batch of input data using the model tokenizer."""
            return tokenizer(batch["sentence"], max_length=128, truncation=True, padding="max_length")
    # Encode the input data
    dataset = dataset.map(encode_batch, batched=True)
    # The transformers model expects the target class column to be named "labels"
    dataset = dataset.rename_column("label", "labels")
    # Transform to pytorch tensors and only output the required columns
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    id2label = task2id2label[task_name]
    label2id = {label: id for id, label in id2label.items()}
    num_labels = len(id2label.keys())

    # train
    config = RobertaConfig.from_pretrained(
        "roberta-base",
        num_labels=num_labels,
    )
    model = RobertaAdapterModel.from_pretrained(
        "WillHeld/roberta-base-" + task_name.lower().replace('-',''),
        config=config,
    )

    # Load the pre-trained adapters we want to fuse
    for adapter_name in adapter_names:
        model.load_adapter(adapters_path + "/" + adapter_name, with_head=False)

    #! Add a null adapter
    model.add_adapter('null')
    #! Add the lexical adapter trained on AAVE
    model.load_adapter(adapters_path + "/mnli_lexical_aave", with_head=False, load_as='mnli_lexical_aave')
    adapter_names += ['null', 'mnli_lexical_aave']

    # Add a fusion layer for all loaded adapters
    adapter_setup = Fuse(*adapter_names)
    model.add_adapter_fusion(adapter_setup)
    fusion_name = ','.join(adapter_names)

    #TODO: Change the fusion_name


    model.set_active_adapters(adapter_setup)

    # Add a classification head for our target task
    model.add_classification_head(task_name, num_labels=len(id2label))

    # Unfreeze and activate fusion setup
    model.train_adapter_fusion(adapter_setup)

    ##! count parameter number
    print('total number: ' + str(sum(torch.numel(p) for p in model.parameters())))

    trainable_model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    trainable_params_num = sum([np.prod(p.size()) for p in trainable_model_parameters])
    print('trainable number: ' + str(trainable_params_num))
    ##!

    training_args = TrainingArguments(
        learning_rate=learning_rate,
        num_train_epochs=train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_steps=1000,
        save_steps=1000,
        evaluation_strategy= "steps",
        save_strategy="steps",
        save_total_limit=2,
        output_dir= output_path + "/log",
        overwrite_output_dir=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_acc",
        remove_unused_columns=False, # This line is important to ensure the dataset labels are properly passed to the model

        local_rank=int(os.environ.get('LOCAL_RANK', -1)),
        save_on_each_node=True,

        report_to='wandb',
        run_name=task_name + "_" + str(args.job_id),

        fp16=True,
        disable_tqdm=False,
        auto_find_batch_size=True,
    )

    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"] if task_name != 'mnli' else dataset["dev_matched"],
        compute_metrics=compute_accuracy,
    )

    torch.cuda.empty_cache()

    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.save_model()
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    # test
    for dialect in test_dialects:
        split = 'dev' if task_name != 'mnli' else 'dev_mismatched'
        if dialect == "VALUE":
            split = 'validation_mismatched' if task_name == 'mnli' else split
            test_dataset = load_dataset('SALT-NLP/' + task_name + '_' + dialect, split= split)
        elif dialect == "GLUE":
            split = 'validation_mismatched' if task_name == 'mnli' else split
            test_dataset = load_dataset('glue', task_name, split= split)
        else:
            test_dataset = load_dataset('liuyanchen1015/' + task_name + '_' + dialect, split= split)
        test_dataset = test_dataset.map(encode_batch, batched=True)
        # The transformers model expects the target class column to be named "labels"
        test_dataset = test_dataset.rename_column("label", "labels")
        # Transform to pytorch tensors and only output the required columns
        test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        metrics = trainer.evaluate(test_dataset)
        trainer.log_metrics(dialect + '_' + split, metrics)
        trainer.save_metrics(dialect + '_' + split, metrics)

    # test
    for dialect in test_dialects:
        split = 'dev' if task_name != 'mnli' else 'dev_matched'
        if dialect == "VALUE":
            split = 'validation_matched' if task_name == 'mnli' else split
            test_dataset = load_dataset('SALT-NLP/' + task_name + '_' + dialect, split= split)
        elif dialect == "GLUE":
            split = 'validation_matched' if task_name == 'mnli' else split
            test_dataset = load_dataset('glue', task_name, split= split)
        else:
            test_dataset = load_dataset('liuyanchen1015/' + task_name + '_' + dialect, split= split)
        test_dataset = test_dataset.map(encode_batch, batched=True)
        # The transformers model expects the target class column to be named "labels"
        test_dataset = test_dataset.rename_column("label", "labels")
        # Transform to pytorch tensors and only output the required columns
        test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        metrics = trainer.evaluate(test_dataset)
        trainer.log_metrics(dialect + '_' + split, metrics)
        trainer.save_metrics(dialect + '_' + split, metrics)

    # predict
    for dialect in predict_dialects:
        split = 'test' if task_name != 'mnli' else 'test_matched'
        if dialect == "VALUE":
            predict_dataset = load_dataset('SALT-NLP/' + task_name + '_' + dialect, split= split)
        elif dialect == "GLUE":
            predict_dataset = load_dataset('glue', task_name, split= split)
        else:
            predict_dataset = load_dataset('liuyanchen1015/' + task_name + '_' + dialect, split= split)

        predict_dataset = predict_dataset.map(encode_batch, batched=True)
        # Transform to pytorch tensors and only output the required columns
        predict_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

        predictions = trainer.predict(predict_dataset, metric_key_prefix="predict").predictions
        predictions = np.argmax(predictions, axis=1)

        output_predict_file = os.path.join(training_args.output_dir, f"predict_results_{task_name}_{dialect}.tsv")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                print(f"***** Predict results {dialect} *****")
                writer.write("index\tprediction\n")
                for index, item in enumerate(predictions):
                    if task_name!= 'sst2':
                        item = id2label[item]
                    writer.write(f"{index}\t{item}\n")

    # save the model
    if torch.distributed.get_rank() == 0:
        model.save_adapter_fusion(output_path + "/saved", adapter_setup, with_head=True)
        model.save_all_adapters(output_path + "/saved")
