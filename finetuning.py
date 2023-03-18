from huggingface_hub import login
import numpy as np
from datasets import load_dataset
import transformers
from transformers import RobertaTokenizer, RobertaConfig, RobertaForSequenceClassification
from transformers import TrainingArguments, Trainer, EvalPrediction
import argparse

login('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')
transformers.logging.set_verbosity_error()

task2id2label={
    'sst2': { 0: "negative", 1: "positive"},
    'mnli' :{0: 'entailment', 1: 'neutral', 2: 'contradiction'},
}

def compute_accuracy(p: EvalPrediction):
  preds = np.argmax(p.predictions, axis=1)
  return {"acc": (preds == p.label_ids).mean()}

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', default='MNLI')
    parser.add_argument('--train_data', default='None')
    args = parser.parse_args()

    # setting
    train_data = args.train_data
    task_name = args.task_name

    task_name = task_name.lower().replace('-','')
    data_path = 'liuyanchen1015/' + task_name + '_' + train_data

    output_dir = "./outputs/finetuning/" + task_name + '_' + train_data

    # data preprocessing
    dataset = load_dataset(data_path)

    dataset['dev'] = dataset['validation_matched'] if task_name == 'mnli' else dataset['validation']
    dataset['test'] = dataset['validation_mismatched'] if task_name == 'mnli' else dataset['test']

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

    # training
    config = RobertaConfig.from_pretrained(
      "roberta-base",
      num_labels=num_labels,
    )
    model = RobertaForSequenceClassification.from_pretrained(
        "WillHeld/roberta-base-" + task_name,
        config=config,
    )

    training_args = TrainingArguments(
        learning_rate=2e-5,
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        logging_steps=2000,
        save_steps=2000,
        evaluation_strategy="steps",
        save_strategy="steps",
        output_dir=output_dir,
        overwrite_output_dir=True,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_acc",
        # The next line is important to ensure the dataset labels are properly passed to the model
        remove_unused_columns=False,
        push_to_hub=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["dev"],
        compute_metrics=compute_accuracy,
    )

    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.save_model()
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    test_dialects = ['MULTI', 'AppE', 'ChcE', 'CollSgE', 'IndE', 'VALUE', 'GLUE']

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

