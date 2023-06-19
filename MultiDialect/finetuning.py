import os
from huggingface_hub import login
import numpy as np
from datasets import load_dataset
import transformers
from transformers import RobertaTokenizer, RobertaConfig, RobertaForSequenceClassification, RobertaAdapterModel
from transformers import TrainingArguments, Trainer, EvalPrediction, AdapterTrainer, set_seed
import argparse

set_seed(42)
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
    parser.add_argument('--job_id', default='2')
    # parser.add_argument("--learning_rate", default=2e-5)
    parser.add_argument("--train_epochs", default=5)
    parser.add_argument("--batch_size", default=64)
    parser.add_argument("--adapter_tuning", default=False)
    args = parser.parse_args()

    # setting
    train_data = args.train_data
    task_name = args.task_name
    task_name = task_name.lower().replace('-','')
    adapter_tuning = args.adapter_tuning
    learning_rate = 0.0003 if adapter_tuning else 2e-5
    train_epochs = args.train_epochs
    batch_size = args.batch_size

    output_dir = './outputs/' + ('finetuning/' if not adapter_tuning else 'adapter_tuning/') + task_name + '_' + train_data \
        + '/' + str(learning_rate) + '_' + str(batch_size) + '_' + str(train_epochs) + '/' + str(args.job_id)

    # data preprocessing
    if train_data == "VALUE":
        dataset = load_dataset('SALT-NLP/' + task_name + '_VALUE')
        dataset['dev'] = dataset['validation_matched'] if task_name == 'mnli' else dataset['validation']
    else:
        dataset = load_dataset('liuyanchen1015/' + task_name + '_' + train_data)
        dataset['dev'] = dataset['dev_matched'] if task_name == 'mnli' else dataset['dev']

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

    if adapter_tuning:
        model = RobertaAdapterModel.from_pretrained(
            "WillHeld/roberta-base-" + task_name,
            config=config,
        )

        adapter_name = task_name + '_' + train_data
        # Add a new adapter
        model.add_adapter(adapter_name)
        # Add a matching classification head
        model.add_classification_head(
            adapter_name,
            num_labels=num_labels,
            id2label=id2label
            )
        # Activate the adapter
        model.train_adapter(adapter_name)

    else:
        model = RobertaForSequenceClassification.from_pretrained(
        "WillHeld/roberta-base-" + task_name,
        config=config,
        )

    training_args = TrainingArguments(
        learning_rate=learning_rate,
        num_train_epochs=train_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
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
        push_to_hub=True if not adapter_tuning else False,
        hub_model_id='liuyanchen1015/roberta-base-' + task_name + '_' + train_data
    )

    trainer_class = AdapterTrainer if adapter_tuning else Trainer

    trainer = trainer_class(
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

    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)






    test_dialects = ['MULTI', 'AppE', 'ChcE', 'CollSgE', 'IndE', 'VALUE', 'GLUE']
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


    predict_dialects = ['MULTI', 'AppE', 'ChcE', 'CollSgE', 'IndE', 'VALUE', 'GLUE']
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

    if adapter_tuning:
        model.push_adapter_to_hub(repo_name= 'pfadapter-roberta-base-' + adapter_name, adapter_name=adapter_name)
