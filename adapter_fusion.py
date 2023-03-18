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
from transformers import RobertaTokenizer, RobertaConfig, RobertaAdapterModel, TrainingArguments, AdapterTrainer, EvalPrediction
from huggingface_hub import login
from torch.utils.data import DataLoader
from scipy.special import softmax

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
    parser.add_argument('--task_name', default='MNLI')
    parser.add_argument('--transformation_rules_path', default='../tmp/transformation_rules_list.json')
    parser.add_argument('--job_id', default='2')
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    # setting
    task_name = args.task_name
    task_name = task_name.lower().replace('-','')
    transformation_rules_path = args.transformation_rules_path

    adapter_names = json.load(open(transformation_rules_path, 'r'))
    adapter_names = [task_name + "_" + section for section in adapter_names]

    output_path = "./Fusion/outputs/" + task_name + "_" + str(args.job_id)
    fig_output_dir = os.path.join(output_path, 'coefplots/')
    adapters_path = "./Adapters"

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
        learning_rate=5e-5,
        num_train_epochs=5,
        per_device_train_batch_size=6,
        per_device_eval_batch_size=6,
        logging_steps=4000,
        save_steps=4000,
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

    # # analysis
    # print('Analyzing...')
    # if not os.path.exists(output_path + "/layer2scores.json"):

    #     model.eval()

    #     analysis_dataset = dataset[analysis_split]
    #     if "labels" in analysis_dataset.column_names:
    #         analysis_dataset = analysis_dataset.remove_columns("labels")
    #     test_dataloader = DataLoader(analysis_dataset, batch_size=training_args.per_device_eval_batch_size)

    #     with torch.no_grad():
    #         layer2max_scores=defaultdict(list)
    #         layer2avg_scores=defaultdict(list)

    #         for batch in tqdm(test_dataloader):
    #             batch = {k: v.to(device) for k, v in batch.items()}

    #             outputs = model(**batch, output_adapter_fusion_attentions=True)
    #             attention_scores = outputs.adapter_fusion_attentions

    #             for layer in range(12):
    #                 scores = attention_scores[fusion_name][layer]['output_adapter'].max(axis=1)
    #                 scores = softmax(scores, axis=1)
    #                 layer2max_scores[layer] += scores.tolist()

    #                 scores = attention_scores[fusion_name][layer]['output_adapter'].mean(axis=1)
    #                 scores = softmax(scores, axis=1)
    #                 layer2avg_scores[layer] += scores.tolist()

    #     layer2scores = {'max': layer2max_scores, 'avg': layer2avg_scores}

    #     with open(output_path + "/layer2scores.json", "w+") as outfile:
    #         json.dump(layer2scores, outfile)

    # else:
    #     with open(output_path + "/layer2scores.json", 'r') as infile:
    #         layer2scores = json.load(infile)




    # tagged_dataset = []
    # file_name = analysis_split.replace('validation', 'dev') if task_name=='mnli' and 'validation' in analysis_split else analysis_split
    # with open('../Value/value_code/data/VALUE/' + task_name + '/' + file_name + '.tsv', 'r') as f:
    #     next(f)
    #     if task_name == 'mnli':
    #             for line in f:
    #                 # index, promptID, pairID, genre, sentence1_binary_parse, sentence2_binary_parse, sentence1_parse, sentence2_parse, sentence1_glue, sentence1_glue_html, sentence1, sentence2_glue, sentence2_glue_html, sentence2, sentence1_ass, sentence1_been_done, sentence1_dey_it, sentence1_drop_aux, sentence1_got, sentence1_lexical, sentence1_negative_concord, sentence1_negative_inversion, sentence1_null_genetive, sentence1_null_relcl, sentence1_total, sentence1_uninflect, sentence2_ass, sentence2_been_done, sentence2_dey_it, sentence2_drop_aux, sentence2_got, sentence2_lexical, sentence2_negative_concord, sentence2_negative_inversion, sentence2_null_genetive, sentence2_null_relcl, sentence2_total, sentence2_uninflect = line.strip().split('\t')
    #                 # index, promptID, pairID, genre, sentence1_binary_parse, sentence2_binary_parse, sentence1_parse, sentence2_parse, sentence1-glue, sentence1-glue-html, sentence1, sentence2-glue, sentence2-glue-html, sentence2, label1, label2, label3, label4, label5, gold_label, sentence1-ass, sentence1-been_done, sentence1-dey_it, sentence1-drop_aux, sentence1-got, sentence1-lexical, sentence1-negative_concord, sentence1-negative_inversion, sentence1-null_genetive, sentence1-null_relcl, sentence1-total, sentence1-uninflect, sentence2-ass, sentence2-been_done, sentence2-dey_it, sentence2-drop_aux, sentence2-got, sentence2-lexical, sentence2-negative_concord, sentence2-negative_inversion, sentence2-null_genetive, sentence2-null_relcl, sentence2-total, sentence2-uninflect = line.strip().split('\t')
    #                 index, promptID, pairID, genre, sentence1_binary_parse, sentence2_binary_parse, sentence1_parse, sentence2_parse, sentence1_glue, sentence1_glue_html, sentence1, sentence2_glue, sentence2_glue_html, sentence2, label1, gold_label, sentence1_ass, sentence1_been_done, sentence1_dey_it, sentence1_drop_aux, sentence1_got, sentence1_lexical, sentence1_negative_concord, sentence1_negative_inversion, sentence1_null_genetive, sentence1_null_relcl, sentence1_total, sentence1_uninflect, sentence2_ass, sentence2_been_done, sentence2_dey_it, sentence2_drop_aux, sentence2_got,	sentence2_lexical, sentence2_negative_concord, sentence2_negative_inversion, sentence2_null_genetive, sentence2_null_relcl, sentence2_total, sentence2_uninflect = line.strip().split('\t')

    #                 ex = {}
    #                 # ex['index'] = index
    #                 ex['ass'] = int(sentence1_ass + sentence2_ass)
    #                 ex['been_done'] = int(sentence1_been_done + sentence2_been_done)
    #                 ex['dey_it'] = int(sentence1_dey_it + sentence2_dey_it)
    #                 ex['drop_aux'] = int(sentence1_drop_aux + sentence2_drop_aux)
    #                 ex['got'] = int(sentence1_got + sentence2_got)
    #                 ex['lexical'] = int(sentence1_lexical + sentence2_lexical)
    #                 ex['negative_concord'] = int(sentence1_negative_concord + sentence2_negative_concord)
    #                 ex['negative_inversion'] = int(sentence1_negative_inversion + sentence2_negative_inversion)
    #                 ex['null_genetive'] = int(sentence1_null_genetive + sentence2_null_genetive)
    #                 ex['null_relcl'] = int(sentence1_null_relcl + sentence2_null_relcl)
    #                 ex['uninflect'] = int(sentence1_uninflect + sentence2_uninflect)

    #                 tagged_dataset.append(ex)

    #     else:
    #         for line in f:
    #             index, glue, html, sentence, ass, been_done, dey_it, drop_aux, got, lexical, negative_concord, negative_inversion, null_genetive, null_relcl, total, uninflect = line.strip().split('\t')

    #             ex = {}
    #             # ex['index'] = index
    #             ex['ass'] = int(ass)
    #             ex['been_done'] = int(been_done)
    #             ex['dey_it'] = int(dey_it)
    #             ex['drop_aux'] = int(drop_aux)
    #             ex['got'] = int(got)
    #             ex['lexical'] = int(lexical)
    #             ex['negative_concord'] = int(negative_concord)
    #             ex['negative_inversion'] = int(negative_inversion)
    #             ex['null_genetive'] = int(null_genetive)
    #             ex['null_relcl'] = int(null_relcl)
    #             ex['uninflect'] = int(uninflect)

    #             tagged_dataset.append(ex)

    # print(tagged_dataset)


    # def relevant_score_analysis(layer2scores, layer, mode='weighted', metric='max'):
    #     scores = layer2scores[metric][layer]
    #     assert(len(scores) == len(tagged_dataset))

    #     sections = json.load(open(transformation_rules_path, 'r'))
    #     section2dist = dict()
    #     for section in sections:
    #         section2dist[section] = {i:0 for i in range(len(adapter_names))}
    #     section2dist['overall'] = {i:0 for i in range(len(adapter_names))}

    #     if mode == 'weighted':
    #         # consider all weights
    #         for weights, ex in zip(scores, tagged_dataset):
    #             for i in range(len(weights)):
    #                 section2dist['overall'][i] += weights[i]

    #         for section, value in ex.items():
    #                 if value != 0:
    #                     for i in range(len(weights)):
    #                         section2dist[section][i] += weights[i]
    #     elif mode == 'dominate':
    #         # dominate adapter id
    #         for weights, ex in zip(scores, tagged_dataset):
    #             i = np.argmax(weights)
    #             section2dist['overall'][i] += 1

    #             for section, value in ex.items():
    #                     if value != 0:
    #                         section2dist[section][i] += 1
    #     else:
    #         raise NotImplementedError('Haven\'t implemented this!')


    #     for section in section2dist.keys():
    #         total_num = sum(section2dist[section].values())
    #         for k, v in section2dist[section].items():
    #             section2dist[section][k] = v / total_num


    #     dataframe = pd.DataFrame.from_dict(section2dist).T
    #     coeff = dataframe.apply(lambda x: round(x, 4) * 100).rename(columns={ i:adapter_name for i, adapter_name in enumerate(adapter_names)})

    #     last_row = coeff.iloc[-1]
    #     coeff = coeff.apply(lambda row: row - last_row, axis=1)
    #     coeff = coeff.iloc[:-1 , :]


    #     plt.figure(figsize=(16, 13), dpi=50)
    #     sns.set(font_scale=1.6)
    #     # heatmap = sns.heatmap(coeff, vmin=-7, vmax=7, annot=True, cmap=sns.cubehelix_palette(start=.5, rot=-.75, as_cmap=True), fmt='.4g')
    #     heatmap = sns.heatmap(coeff, vmin=-7, vmax=7, annot=True, cmap='RdBu', fmt='.4g')
    #     fig_name = 'Layer ' + str(layer) +' (' + mode + ', '+ metric + ')'
    #     heatmap.set_title(fig_name, fontdict={'fontsize':18}, pad=12);

    #     plt.savefig(os.path.join(fig_output_dir, fig_name + '.png'), dpi=120)

    # if not os.path.isdir(fig_output_dir):
    #     os.mkdir(fig_output_dir)

    # for layer in range(12):
    #     # relevant_score_analysis(layer2scores, layer, 'dominate', 'max')
    #     # relevant_score_analysis(layer2scores, layer, 'dominate', 'avg')

    #     relevant_score_analysis(layer2scores, layer, 'weighted', 'max')
    #     # relevant_score_analysis(layer2scores, layer, 'weighted', 'avg')
