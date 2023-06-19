from huggingface_hub import login
import numpy as np
from datasets import load_dataset
import transformers
from transformers import RobertaTokenizer, RobertaConfig, RobertaModelWithHeads
from transformers import TrainingArguments, AdapterTrainer, EvalPrediction
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
    parser.add_argument('--transformation_rule', default='None')
    args = parser.parse_args()

    # setting
    transformation_rule = args.transformation_rule
    task_name = args.task_name

    task_name = task_name.lower().replace('-','')
    adapter_name = task_name + '_' + transformation_rule
    data_path = 'liuyanchen1015/MULTI_VALUE_' + adapter_name
    original_data_path = 'SALT-NLP/' + task_name +'_VALUE'

    output_dir = "./AdapterTuning"
    save_dir = "./Adapters"
    num_steps = 10000
    batch_size = 64

    # data preprocessing
    print('Training adapter {}...\n'.format(adapter_name))

    dataset = load_dataset(data_path)
    original_dataset = load_dataset(original_data_path)

    dataset['dev'] = original_dataset['validation_matched'] if task_name == 'mnli' else original_dataset['validation']
    dataset['test'] = original_dataset['validation_mismatched'] if task_name == 'mnli' else original_dataset['test']

    num_train_examples = len(dataset['train'])
    num_epochs = num_steps * batch_size / num_train_examples

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
    model = RobertaModelWithHeads.from_pretrained(
        "WillHeld/roberta-base-" + task_name,
        config=config,
    )

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

    training_args = TrainingArguments(
        learning_rate=0.0003,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_steps=num_steps/10,
        save_steps=num_steps/10,
        evaluation_strategy="steps",
        save_strategy="steps",
        output_dir=output_dir + '/' + adapter_name + '/',
        overwrite_output_dir=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_acc",
        # The next line is important to ensure the dataset labels are properly passed to the model
        remove_unused_columns=False,
    )

    trainer = AdapterTrainer(
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

    model.save_adapter(save_dir + '/' + adapter_name, adapter_name)