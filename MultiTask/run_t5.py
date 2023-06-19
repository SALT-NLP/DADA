#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import sys
import json
import torch
from dataclasses import dataclass, field
from typing import Optional, List
from random import uniform

import datasets
import numpy as np
from datasets import load_dataset, concatenate_datasets, DatasetDict
from torch.utils.data import DataLoader

import evaluate
import transformers
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    EvalPrediction,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed,
)
from transformers.adapters import AdapterArguments, setup_adapter_training
from transformers.adapters.composition import Fuse
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from huggingface_hub import login
import nvidia_smi

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.26.0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/text-classification/requirements.txt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#! GPU Info
_GPU = False
_NUMBER_OF_GPU = 0

def _check_gpu():
    global _GPU
    global _NUMBER_OF_GPU
    nvidia_smi.nvmlInit()
    _NUMBER_OF_GPU = nvidia_smi.nvmlDeviceGetCount()
    if _NUMBER_OF_GPU > 0:
        _GPU = True

def _print_gpu_usage(detailed=False):

    if not detailed:
        for i in range(_NUMBER_OF_GPU):
            handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
            info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
            print(f'GPU-{i}: GPU-Memory: {_bytes_to_megabytes(info.used)}/{_bytes_to_megabytes(info.total)} MB')

def _bytes_to_megabytes(bytes):
    return round((bytes/1024)/1024,2)
#!

canonical_task_name = {
    "cola": 'CoLA',
    "mnli": 'MNLI',
    "mrpc": 'MRPC',
    "qnli": 'QNLI',
    "qqp": 'QQP',
    "rte": 'RTE',
    "sst2": 'SST-2',
    "stsb": 'STS-B',
    "wnli": 'WNLI',
}

decanonical_task_name = {v:k for k,v in canonical_task_name.items()}

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

task_to_id_to_label={
    'cola': { 0: 'unacceptable', 1:'acceptable'},
    'mnli' :{0: 'entailment', 1: 'neutral', 2: 'contradiction'},
    'mrpc': {0: "not_equivalent", 1: "equivalent"},
    'qnli': {0: 'entailment', 1: 'not_entailment'},
    'qqp': {0: 'not_duplicate', 1: 'duplicate'},
    'rte': {0: 'entailment', 1: 'not_entailment'},
    'sst2': { 0: "negative", 1: "positive"},
    'stsb': {},
    'wnli': {0: 'not_entailment', 1: 'entailment'},
}

task_to_templates={
    "cola": [
        ("Sentence: \"{sentence}\"\nPick from options: would a linguist rate "
         "this sentence to be acceptable linguistically?\n\n{options_}...I "
         "think the answer is", "{answer}"),
        ("{sentence}\n\nHow would you consider the linguistic integrity of the"
         " preceding sentence?\n{options_}", "{answer}"),
        ("Test sentence: \"{sentence}\"\nIs this test sentence a correct "
         "grammatical English sentence?\n\n{options_}", "{answer}"),
        ("Sentence: \"{sentence}\"\nWould a linguist rate this sentence to be "
         "acceptable linguistically?\n\n{options_}", "{answer}"),
        ("Choose from options, is the following sentence linguistically "
         "acceptable?\n{sentence}\n{options_}", "{answer}"),
        ("Choose from the possible answers, would the following sentence, by "
         "the strictest standards, be considered correct by a "
         "linguist?\n\n{sentence}\n{options_}", "{answer}"),
        ("Multi-choice problem: Is the next sentence syntactically and "
         "semantically acceptable?\n\n{sentence}\n{options_}", "{answer}"),
        ("Would a linguist find the following sentence to be a valid English "
         "sentence grammatically?\n\n{sentence}\n{options_}", "{answer}"),
        ("Generate short a sentence that can be linguistically classified as "
         "{answer} ({options_})", "{sentence}"),
        ("Produce a brief English sentence that would be considered "
         "grammatically as category: {answer}\nAll categories: {options_}",
         "{sentence}"),
         ("cola sentence: {sentence}", "{answer}")
    ],
    "mnli": [
        ("Premise: {premise}\n\nHypothesis: {hypothesis}\n\nDoes the premise "
         "entail the hypothesis?\n\n{options_}", "{answer}"),
        ("Premise: {premise}\nHypothesis: {hypothesis}\nIs the hypothesis "
         "entailed by the premise?\n{options_} And the answer is:", "{answer}"),
        ("Here is a premise:\n{premise}\n\nHere is a "
         "hypothesis:\n{hypothesis}\n\nHere are the options: {options_}\nIs it"
         " possible to conclude that if the premise is true, then so is the "
         "hypothesis?\n", "{answer}"),
        ("Sentence 1: {premise}\n\nSentence 2: {hypothesis}\n{options_}\nIs "
         "this second sentence entailed by the first sentence?\n\n",
         "{answer}"),
        ("See the multi-choice question below:\n\nSentence 1: "
         "{premise}\n\nSentence 2: {hypothesis}\n\nIf the first sentence is "
         "true, then is the second sentence true?\n{options_}", "{answer}"),
        ("Based on the premise \"{premise}\", can we conclude the hypothesis "
         "\"{hypothesis}\" is true (see options)?\n\n{options_}", "{answer}"),
        ("Choose your answer from options. Premise: \"{premise}\" If this "
         "premise is true, what does that tell us about whether it entails the"
         " hypothesis \"{hypothesis}\"?\n\n{options_}", "{answer}"),
        ("Premise:\n\"{premise}\" Based on this premise, is the hypothesis "
         "\"{hypothesis}\" true?\n{options_}", "{answer}"),
        ("If {premise}, can we conclude that \"{hypothesis}\"?\n{options_}",
         "{answer}"),
        ("{premise}\n\nDoes it follow that \"{hypothesis}\"?\n{options_}",
         "{answer}"),
         ("mnli hypothesis: {hypothesis} premise: {premise}", "{answer}")
    ],
    "mrpc": [
        ("Here are two sentences:\n{sentence1}\n{sentence2}\nDo they have the "
         "same meaning?\n{options_}", "{answer}"),
        ("Here are two sentences:\n\n{sentence1}\n\n{sentence2}\nChoose your "
         "answer: are the two sentences saying the same thing?\n{options_}",
         "{answer}"),
        ("{sentence1}\n\n{sentence2}\n\nSelect from the options at the end. Do"
         " the above sentences mean the same thing?\n{options_}", "{answer}"),
        ("{sentence1}\n\n{sentence2}\n\nPlease tell me if the sentences above "
         "mean the same.\n{options_}", "{answer}"),
        ("{sentence1}\n{sentence2}\nSelect from the options at the end. Are "
         "these sentences conveying the same meaning?\n{options_}", "{answer}"),
        ("{sentence1}\n{sentence2}\n(See options at the end). If the first "
         "sentence is true, is the second one also true?\n{options_}",
         "{answer}"),
        ("{sentence1}\n{sentence2}\nAre these two sentences paraphrases of "
         "each other?\n{options_}", "{answer}"),
        ("Do the following two sentences have the same "
         "meaning?\n{sentence1}\n{sentence2}\n\n{options_}\nThe answer is:",
         "{answer}"),
        ("Do these two sentences mean the same "
         "thing?\n{sentence1}\n{sentence2}\n\n{options_}...I think the answer "
         "is", "{answer}"),
        ("Do these sentences have the same "
         "meaning?\n{sentence1}\n{sentence2}\n\n{options_}", "{answer}"),
         ("mrpc sentence1: {sentence1} sentence2: {sentence2}", "{answer}")
    ],
    "qnli": [
        ("Does the sentence \"{sentence}\" answer the question "
         "\"{question}\"\n\n{options_}", "{answer}"),
        ("Single/multi-select question: Does the sentence \"{sentence}\" "
         "provide a valid answer to the question \"{question}\"\n{options_}",
         "{answer}"),
        ("Choose your answer: Is \"{sentence}\" a good answer to the question "
         "\"{question}\"\n{options_}", "{answer}"),
        ("{options_}\nDoes \"{sentence}\" correctly answer the question of "
         "{question}\n\n", "{answer}"),
        ("Choose your reply from the options at the end. Does \"{sentence}\" "
         "contain the correct answer to \"{question}\"\n{options_}",
         "{answer}"),
        ("Q: {question}\n A: {sentence}\n Does the answer correctly answer the"
         " question\n\n{options_}", "{answer}"),
        ("Question: {question}\nAnswer: {sentence}\n A single-select problem: "
         "Is the question answered in a satisfactory fashion?\n\n{options_}",
         "{answer}"),
        ("Question: {question}\n\nIs {sentence} a good answer to this "
         "question?\n\n{options_}", "{answer}"),
        ("Question: {question}\n\nIs \"{sentence}\" the correct answer?\n"
         "{options_}", "{answer}"),
        ("Can you generate a question with a factual answer?", "{question}"),
        ("qnli question: {question} sentence: {sentence}", "{answer}")
    ],
    "qqp": [
        ("{question1}\n{question2}\nMulti-choice problem: Would you say that "
         "these questions are the same?\n{options_}", "{answer}"),
        ("{question1}\n{question2}\nDo those questions have the same "
         "meaning?\n{options_}", "{answer}"),
        ("{question1}\n{question2}\n\nMulti-choice problem: Are these two "
         "questions inquiring about the same information?\n{options_}",
         "{answer}"),
        ("{question1}\n\n{question2}\n\nPlease tell me if those questions are "
         "the same.\n{options_}", "{answer}"),
        ("{question1}\n\n{question2}\n\nChoose your answer. Are these two "
         "questions paraphrases of each other?\n{options_}", "{answer}"),
        ("First question: {question1}\nSecond question: {question2}\nAre these"
         " two questions asking the same thing?\n{options_}", "{answer}"),
        ("Question 1: {question1}\nQuestion 2: {question2}\n{options_}\nAre "
         "questions 1 and 2 asking the same thing?", "{answer}"),
        ("Question 1: {question1}\nQuestion 2: {question2}\n{options_}\nWould "
         "the answer to these two questions be the same?", "{answer}"),
        ("Choose from the options at the end. Are the following two questions "
         "the same?\n{question1}\n{question2}\n\n{options_}\nThe answer is:",
         "{answer}"),
        ("Do these questions have the same "
         "meaning?\n{question1}\n{question2}\n\n{options_}", "{answer}"),
         ("qqp question1: {question1} question2: {question2}", "{answer}")
    ],
    "rte": [
        ("{premise}\n\nQuestion with options: Based on the paragraph above can"
         " we conclude that \"{hypothesis}\"?\n\n{options_}", "{answer}"),
        ("{premise}\n\nBased on that paragraph can we conclude that the "
         "sentence below is true?\n{hypothesis}\n\n{options_}", "{answer}"),
        ("{premise}\n\nQ with options: Can we draw the following "
         "conclusion?\n{hypothesis}\n\n{options_}", "{answer}"),
        ("{premise}\nDoes this next sentence follow, given the preceding "
         "text?\n{hypothesis}\n\n{options_}", "{answer}"),
        ("{premise}\n{options_}\nQuestion: Can we infer the "
         "following?\n{hypothesis}", "{answer}"),
        ("Read the following paragraph and determine if the hypothesis is "
         "true. Select from options at the end:\n\n{premise}\n\nHypothesis: "
         "{hypothesis}\n{options_}\nThe answer is", "{answer}"),
        ("Read the text and determine if the sentence is "
         "true:\n\n{premise}\n\nSentence: {hypothesis}\n{options_}\nA:",
         "{answer}"),
        ("Question with options: can we draw the following hypothesis from the"
         " context? \n\nContext:\n\n{premise}\n\nHypothesis: "
         "{hypothesis}\n{options_}\nA:", "{answer}"),
        ("Determine if the sentence is true based on the text below. Choose "
         "from options.\n{hypothesis}\n\n{premise}\n{options_}", "{answer}"),
        ("Generate a context and a hypothesis.",
         "Context: {premise}\n\nHypothesis: {hypothesis}"),
         ("rte sentence1: {sentence1} sentence2: {sentence2}", "{answer}")
    ],
    "sst2": [
        ("Review:\n{sentence}\nIs this movie review sentence negative or "
         "positive?\n{options_}\nThe answer is:", "{answer}"),
        ("{options_}\nShort movie review: {sentence}\nDid the critic thinking "
         "positively or negatively of the movie?\n\n", "{answer}"),
        ("Sentence from a movie review: {sentence}\nSelect your answer: was "
         "the movie seen positively or negatively based on the preceding "
         "review?\n\n{options_}", "{answer}"),
        ("\"{sentence}\"\nHow would the sentiment of this sentence be "
         "perceived --\n\n{options_}\nAnswer:", "{answer}"),
        ("Is the sentiment of the following sentence positive or negative (see"
         " options at the end)?\n{sentence}\n{options_}", "{answer}"),
        ("What is the sentiment of the following movie (choose your answer "
         "from the options) review sentence?\n{sentence}\n{options_}\nThe "
         "answer is:", "{answer}"),
        ("{options_}Would the following phrase be considered positive or "
         "negative?\n\n{sentence}\n", "{answer}"),
        ("Does the following review have a positive or negative opinion of the"
         " movie?\n\n{sentence}\n{options_}", "{answer}"),
        ("Write a \"{answer}\" movie review ({options_}).", "{sentence}"),
        ("Generate a short movie review that has \"{answer}\" sentiment "
         "({options_}).", "{sentence}"),
         ("sst2 sentence: {sentence}", "{answer}")
    ],
    "stsb": [
        ("{sentence1}\n{sentence2}\n\nRate the textual similarity of these two"
         " sentences on a scale from 0 to 5, where 0 is \"no meaning overlap\""
         " and 5 is \"means the same thing\".\n\n{options_}", "{answer}"),
        ("{sentence1}\n{sentence2}\n\nOn a scale from 0 to 5, where 0 is \"no "
         "meaning overlap\" and 5 is \"means the same thing\", how closely "
         "does the first sentence resemble the second one?\n\n{options_}",
         "{answer}"),
        ("Sentence 1: {sentence1}\n\n Sentence 2: {sentence2}\n\nFrom 0 to 5 "
         "(0=\"no meaning overlap\" and 5=\"means the same thing\"), how "
         "similar are the two sentences?\n\n{options_}", "{answer}"),
        ("Select from options: How similar are the following two "
         "sentences?\n\n{sentence1}\n{sentence2}\n\nGive the answer on a scale"
         " from 0 - 5, where 0 is \"not similar at all\" and 5 is \"means the "
         "same thing\".\n\n{options_}", "{answer}"),
        ("Single/multi-select question: Do the following sentences say the "
         "same thing?\n\n{sentence1}\n{sentence2}\n\nReturn your answer on a "
         "scale from 0 to 5, where 0 is \"not similar\" and 5 is \"very "
         "similar\".\n\n{options_}", "{answer}"),
        ("Rate the similarity of the following two sentences on a scale from 0"
         " to 5, where 0 is \"no meaning overlap\" and 5 is \"means the same "
         "thing\"?\n\n{sentence1}\n{sentence2}\n\n{options_}", "{answer}"),
        ("On a scale from 0-5, where 0 is \"not similar\" and 5 is \"very "
         "similar\", how similar is the sentence \"{sentence1}\" to the "
         "sentence \"{sentence2}\"?\n\n{options_}", "{answer}"),
        ("How similar are these two sentences, on a scale from 0-5 (0 is \"not"
         " similar\" and 5 is \"very "
         "similar\")?\n\n{sentence1}\n{sentence2}\n\n{options_}", "{answer}"),
        ("{sentence1}\n\nGenerate a new sentence that is, on a scale from 0 to"
         " 5, a {answer} in textual similarity to the above sentence.",
         "{sentence2}"),
        ("{sentence2}\n\nWhat is a sentence that would be (on a scale from 0 "
         "to 5) a {answer} out of 5 in terms of textual similarity to the "
         "above sentence?", "{sentence1}"),
         ("stsb sentence1: {sentence1} sentence2: {sentence2}", "{answer}")
    ],
    "wnli": [
        ("If \"{sentence1}\", can we conclude that "
         "\"{sentence2}\"\n{options_}\nI think the answer is", "{answer}"),
        ("If \"{sentence1}\", does it follow that \"{sentence2}\"\n{options_}",
         "{answer}"),
        ("If \"{sentence1}\", is \"{sentence2}\" "
         "correct?\n\n{options_}\nAnswer:", "{answer}"),
        ("Multi-select: Let's say that \"{sentence1}\"\n\nCan we now say that "
         "\"{sentence2}\"?\n\n{options_}", "{answer}"),
        ("\"{sentence1}\" is a true sentence.\n\nDoes this mean that "
         "\"{sentence2}\"?\n\n{options_}", "{answer}"),
        ("Does \"{sentence2}\" appear to be an accurate statement based on "
         "\"{sentence1}\"?\n\n{options_}", "{answer}"),
        ("Can we conclude that \"{sentence2}\" if the statement "
         "\"{sentence1}\" is true?\n\n{options_}", "{answer}"),
        ("Multi-select: Is it possible to draw the conclusion that "
         "\"{sentence2}\" if \"{sentence1}\"?\n\n{options_}", "{answer}"),
        ("Multi-choice problem: Is \"{sentence2}\" true if "
         "\"{sentence1}\"?\n\n{options_}", "{answer}"),
        ("Sentence 1: \"{sentence1}\"\n\n Sentence 2: \"{sentence2}\"\n\nIs "
         "sentence 2 true, based on sentence 1?\n{options_}", "{answer}"),
         ("wnli sentence1: {sentence1} sentence2: {sentence2}", "{answer}")
    ],
}

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    predict_task_names: Optional[List[str]] = field(default=None, metadata={"help": "the list of tasks to predict for"})


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    adapter_fusion: bool = field(
        default=False,
        metadata={"help": "adapter_fusion mode"},
    )
    with_null: bool = field(
        default=False,
        metadata={"help": "whether fuse with an additional null adapter or not"},
    )
    adapters_path: str = field(
        default=None, metadata={"help": "Path to the pretrained adapters"}
    )

    model_name_or_path: str = field(
        default='google/flan-t5-base', metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default='google/flan-t5-base', metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    ignore_mismatched_sizes: bool = field(
        default=False,
        metadata={"help": "Will enable to load a pretrained model whose head dimensions are different."},
    )
    instruction: str = field(
        default="",
        metadata={"help": "The optional natural language instruction to instruct the LM."},
    )



def preprocess_datasets(datasets, task_name, tokenizer, training_args, data_args, padding, max_seq_length, instruction="", truncation=True):
    column_names = list(datasets.values())[0].column_names if isinstance(datasets, DatasetDict) else datasets.column_names
    is_regression = task_name == "stsb"
    sentence1_key, sentence2_key = task_to_keys[task_name]

    if 'value_score' in column_names:
        datasets = datasets.remove_columns(['value_score']) # we don't need this for now (may be changed later)
        column_names.remove('value_score')

    if not is_regression:
        id_to_label = task_to_id_to_label[task_name]

    #! choose which template?
    template, _ = task_to_templates[task_name][0]

    #? with options
    options = ', '.join(id_to_label.values()) if not is_regression else ''
    template = template.replace('{options_}', options)

    #? without options
    # template = template.replace(
    # "\n\n{options_}",
    # "").replace("\n{options_}",
    #             "").replace("{options_}",
    #                         "").replace("\n\n{options_str}",
    #                                     "")

    if instruction:
        template = instruction + '\n'+ template

    def preprocess_function(examples):
        num = len(examples['label'])
        texts = [template] * num

        if sentence2_key is None:
            texts = [text.replace('{' + sentence1_key  + '}', examples[sentence1_key][idx])
                        for idx, text in enumerate(texts)]
        else:
            texts = [text.replace('{' + sentence1_key  + '}', examples[sentence1_key][idx]).replace('{' + sentence2_key  + '}', examples[sentence2_key][idx])
                        for idx, text in enumerate(texts)]
        encoding = tokenizer(texts, padding=padding, max_length=max_seq_length, truncation=truncation, return_tensors="pt")
        input_ids, attention_mask = encoding.input_ids, encoding.attention_mask

        if not is_regression:
            answers = [id_to_label[label_id] if label_id != -1 else '' for label_id in examples['label']]
        else: answers = [str(s) for s in examples['label']]
        target_encoding = tokenizer(answers, padding=padding, max_length=max_seq_length, truncation=truncation, return_tensors="pt")
        labels = target_encoding.input_ids
        labels[labels == tokenizer.pad_token_id] = -100

        result = {'input_ids':input_ids, 'attention_mask': attention_mask, 'labels': labels, 'input': texts, 'output': answers, 'task': [task_name] * num}
        return result

    with training_args.main_process_first(desc="dataset map pre-processing"):
        datasets = datasets.map(
            preprocess_function,
            batched=True,
            load_from_cache_file=not data_args.overwrite_cache,
            desc="Running tokenizer on dataset",
        )

        keep_column_names = ['input_ids', 'attention_mask', 'labels', 'input', 'output']
        remove_column_names = [name for name in column_names if name not in keep_column_names]
        datasets = datasets.remove_columns(remove_column_names)
    return datasets

def main():
    print('Checking for Nvidia GPU:')
    _check_gpu()
    if _GPU:
        _print_gpu_usage()
    else:
        print("No GPU found.")

    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, AdapterArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, adapter_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args, adapter_args = parser.parse_args_into_dataclasses()

    print(model_args.instruction)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model and tokenizer
    tokenizer = T5Tokenizer.from_pretrained(model_args.tokenizer_name)
    model = T5ForConditionalGeneration.from_pretrained(model_args.model_name_or_path, device_map="auto")

    if adapter_args.train_adapter: #? adapter tuning or finetuning
        setup_adapter_training(model, adapter_args, 'multi-task VALUE')
    elif model_args.adapter_fusion: #? adapter fusion mode
        adapter_names = ["been_done", "dey_it", "drop_aux", "got", "lexical", "negative_concord", "negative_inversion", "null_genetive", "null_relcl", "uninflect"]
        # Load the pre-trained adapters we want to fuse
        for adapter_name in adapter_names:
            model.load_adapter(model_args.adapters_path + '-' + adapter_name + '_lr0.001', source="hf", with_head=False)

        if model_args.with_null:
            model.add_adapter('null')
            adapter_names += ['null']

        # Add a fusion layer for all loaded adapters
        adapter_setup = Fuse(*adapter_names)
        model.add_adapter_fusion(adapter_setup)
        fusion_name = ','.join(adapter_names)
        model.set_active_adapters(adapter_setup)
        # Unfreeze and activate fusion setup
        model.train_adapter_fusion(adapter_setup)

    ##! count parameter number
    print('total parameters: ' + str(sum(torch.numel(p) for p in model.parameters())))

    trainable_model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    trainable_params_num = sum([np.prod(p.size()) for p in trainable_model_parameters])
    print('trainable parameters: ' + str(trainable_params_num))
    ##!

    #! Data
    task_names = list(task_to_keys.keys())
    task_names.remove('mrpc') #! No MRPC VALUE Test set
    task_names.remove('wnli') #! T5 not finetuend on WNLI

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # prepare the training datasets
    if training_args.do_train or training_args.do_eval:
        raw_datasets = {}
        for task_name in task_names:
            raw_datasets[task_name] = load_dataset(
                ('SALT-NLP/' if task_name not in ['mrpc', 'wnli'] else 'liuyanchen1015/')
                + task_name + '_VALUE',
                cache_dir=model_args.cache_dir,
                use_auth_token=True if model_args.use_auth_token else None
            ) #* training on VALUE

            # raw_datasets[task_name] = load_dataset('glue', task_name) #* training on SAE GLUE

            raw_datasets[task_name] = preprocess_datasets(raw_datasets[task_name], task_name, tokenizer,
                                                        training_args, data_args, padding, max_seq_length, model_args.instruction, truncation=True)

        train_datasets = []
        eval_datasets = []
        for task_name in task_names:
            train_datasets.append(raw_datasets[task_name]['train'])
            if task_name == 'mnli':
                eval_datasets.append(raw_datasets[task_name]['validation_matched'])
            else:
                eval_datasets.append(raw_datasets[task_name]['validation' if task_name not in ['mrpc', 'wnli'] else 'dev'])

        train_dataset = concatenate_datasets(train_datasets)
        eval_dataset = concatenate_datasets(eval_datasets)

        del raw_datasets, train_datasets, eval_datasets

    if training_args.do_train:
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.shuffle()
            eval_dataset = eval_dataset.select(range(max_eval_samples))

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set:\n{train_dataset[index]['input']}\n{train_dataset[index]['output']}")

    def compute_metrics(eval_arg):
        preds, labels = eval_arg
        # Replace -100
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        # Convert id tokens to text
        text_preds = np.array(tokenizer.batch_decode(preds, skip_special_tokens=True))
        text_labels = np.array(tokenizer.batch_decode(labels, skip_special_tokens=True))

        return {"accuracy": round((text_preds == text_labels).mean().item(), 4)}

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        return_tensors="pt")

    # Initialize our Trainer
    trainer_class = Seq2SeqTrainer
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_train and training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    #! Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    #! Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        eval_datasets = {}
        # prepare the evaluation datasets
        for task_name in task_names:
            if task_name == 'mnli':
                eval_datasets[('VALUE', 'MNLI-m')] = load_dataset(('SALT-NLP/' if task_name not in ['mrpc', 'wnli'] else 'liuyanchen1015/') + task_name + '_VALUE',
                                                                  cache_dir=model_args.cache_dir, use_auth_token=True if model_args.use_auth_token else None,
                                                                  split='validation_matched')
                eval_datasets[('VALUE','MNLI-mm')] = load_dataset(('SALT-NLP/' if task_name not in ['mrpc', 'wnli'] else 'liuyanchen1015/') + task_name + '_VALUE',
                                                                  cache_dir=model_args.cache_dir, use_auth_token=True if model_args.use_auth_token else None,
                                                                  split='validation_mismatched')
                eval_datasets[('GLUE', 'MNLI-m')] = load_dataset('glue', task_name,
                                                                 cache_dir=model_args.cache_dir, use_auth_token=True if model_args.use_auth_token else None,
                                                                 split='validation_matched')
                eval_datasets[('GLUE','MNLI-mm')] = load_dataset('glue', task_name,
                                                                 cache_dir=model_args.cache_dir, use_auth_token=True if model_args.use_auth_token else None,
                                                                 split='validation_mismatched')
            else:
                eval_datasets[('VALUE', canonical_task_name[task_name])] = load_dataset(('SALT-NLP/' if task_name not in ['mrpc', 'wnli'] else 'liuyanchen1015/') + task_name + '_VALUE',
                                                                                        cache_dir=model_args.cache_dir, use_auth_token=True if model_args.use_auth_token else None,
                                                                                        split='validation' if task_name not in ['mrpc', 'wnli'] else 'dev')
                eval_datasets[('GLUE', canonical_task_name[task_name])] = load_dataset('glue', task_name,
                                                                                       cache_dir=model_args.cache_dir, use_auth_token=True if model_args.use_auth_token else None,
                                                                                       split='validation')

        all_metrics = {}
        for (dialect, task_name), eval_dataset in eval_datasets.items():
            de_task_name = decanonical_task_name[task_name.replace('-mm', '').replace('-m', '')]

            id2label = task_to_id_to_label[de_task_name]
            label2id = {label:id for id,label in id2label.items()}

            eval_dataset = preprocess_datasets(eval_dataset, de_task_name, tokenizer,
                                               training_args, data_args, padding, max_seq_length, model_args.instruction, truncation=True)

            #? Log a few random samples from the evaluation set:
            for index in random.sample(range(len(eval_dataset)), 3):
                logger.info(f"Sample {index} of the evaluation set:\n{eval_dataset[index]['input']}\n{eval_dataset[index]['output']}")

            metrics = trainer.evaluate(eval_dataset)
            max_eval_samples = (
                data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            )
            metrics["dialect"] = dialect
            metrics["task"] = task_name
            metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
            trainer.log_metrics(f"{dialect} {task_name} Validation", metrics)

            all_metrics[f"{dialect} {task_name}"] = metrics

        eval_results_file = os.path.join(training_args.output_dir, 'eval.json')
        with open(eval_results_file, 'w') as fout:
            json.dump(all_metrics, fout, indent=4)

        acc_list = []
        for k in all_metrics.keys():
            acc_list.append(all_metrics[k]['eval_accuracy'])
        print('mean: {:.4f}'.format(sum(acc_list) / len(acc_list)))

    #! Prediction
    if training_args.do_predict:
        logger.info("*** Predict ***")

        save_dir = os.path.join(training_args.output_dir, 'predict_results')

        predict_datasets = {}
        # prepare the prediction datasets
        data_args.predict_task_names = data_args.predict_task_names if data_args.predict_task_names else task_names
        for task_name in data_args.predict_task_names:
            if task_name == 'mnli':
                predict_datasets[('VALUE', 'MNLI-m')] = load_dataset(('SALT-NLP/' if task_name not in ['mrpc', 'wnli'] else 'liuyanchen1015/') + task_name + '_VALUE',
                                                                     cache_dir=model_args.cache_dir, use_auth_token=True if model_args.use_auth_token else None,
                                                                     split='test_matched')
                predict_datasets[('VALUE','MNLI-mm')] = load_dataset(('SALT-NLP/' if task_name not in ['mrpc', 'wnli'] else 'liuyanchen1015/') + task_name + '_VALUE',
                                                                     cache_dir=model_args.cache_dir, use_auth_token=True if model_args.use_auth_token else None,
                                                                     split='test_mismatched')
                predict_datasets[('GLUE', 'MNLI-m')] = load_dataset('glue', task_name,
                                                                    cache_dir=model_args.cache_dir, use_auth_token=True if model_args.use_auth_token else None,
                                                                    split='test_matched')
                predict_datasets[('GLUE','MNLI-mm')] = load_dataset('glue', task_name,
                                                                    cache_dir=model_args.cache_dir, use_auth_token=True if model_args.use_auth_token else None,
                                                                    split='test_mismatched')
            else:
                predict_datasets[('VALUE', canonical_task_name[task_name])] = load_dataset(('SALT-NLP/' if task_name not in ['mrpc', 'wnli'] else 'liuyanchen1015/') + task_name + '_VALUE',
                                                                                           cache_dir=model_args.cache_dir, use_auth_token=True if model_args.use_auth_token else None,
                                                                                           split='test')
                predict_datasets[('GLUE', canonical_task_name[task_name])] = load_dataset('glue', task_name,
                                                                                          cache_dir=model_args.cache_dir, use_auth_token=True if model_args.use_auth_token else None,
                                                                                          split='test')

        for (dialect, task_name), predict_dataset in predict_datasets.items():
            if not os.path.exists(f"{save_dir}_{dialect}"):
                os.makedirs(f"{save_dir}_{dialect}")

            de_task_name = decanonical_task_name[task_name.replace('-mm', '').replace('-m', '')]
            is_regression = task_name == 'STS-B'

            id2label = task_to_id_to_label[de_task_name]
            label2id = {label:id for id,label in id2label.items()}

            if data_args.max_predict_samples is not None:
                max_predict_samples = min(len(predict_dataset), data_args.max_predict_samples)
                predict_dataset = predict_dataset.select(range(max_predict_samples))

            predict_dataset = preprocess_datasets(predict_dataset, de_task_name, tokenizer,
                                                  training_args, data_args, padding, max_seq_length, model_args.instruction, truncation=True)

            # all_preds = []
            # predict_dataset = predict_dataset.with_format("torch", device=device)
            # miss = 0
            # for batch in DataLoader(predict_dataset, batch_size=64):
            #     preds = model.generate(batch['input_ids'], max_length=2)

            #     # Convert id tokens to text
            #     preds = np.array(tokenizer.batch_decode(preds, skip_special_tokens=True))

            #     for label in preds:
            #         if label not in label2id.keys():
            #             miss += 1
            #     preds = [label if label in label2id.keys() else random.choice(list(label2id.keys())) for label in preds]

            #     if task_name in ['MNLI-m', 'MNLI-mm', 'QNLI', 'RTE']:
            #         # Convert label to id
            #         preds = np.array([label2id[label] for label in preds])

            #     all_preds.extend(preds)



            preds = trainer.predict(predict_dataset).predictions
            # Convert id tokens to text
            preds = np.array(tokenizer.batch_decode(preds, skip_special_tokens=True))

            miss = 0
            if not is_regression:
                for label in preds:
                    if label not in label2id.keys():
                        miss += 1
                preds = [label if label in label2id.keys() else random.choice(list(label2id.keys())) for label in preds]
            if is_regression:
                tmp_preds = []
                for label in preds:
                    try:
                        tmp_preds.append(float(label))
                    except ValueError:
                        miss += 1
                        tmp_preds.append(uniform(0, 5))
                preds = tmp_preds


            if task_name not in ['MNLI-m', 'MNLI-mm', 'QNLI', 'RTE'] and not is_regression:
                # Convert label to id
                preds = [label2id[label] for label in preds]
            #

            output_predict_file = os.path.join(f"{save_dir}_{dialect}", f"{task_name}.tsv")
            if trainer.is_world_process_zero():
                with open(output_predict_file, "w") as writer:
                    logger.info(f"***** Predict results {dialect} {task_name} *****")
                    logger.info(f"***** Missed Samples {miss} *****")
                    writer.write("index\tprediction\n")
                    for index, item in enumerate(preds):
                        if is_regression:
                            writer.write(f"{index}\t{item:3.3f}\n")
                        else:
                            writer.write(f"{index}\t{item}\n")



    if adapter_args.train_adapter:
        model.push_adapter_to_hub(
            "pfadapter-FLAN-T5-base-multi-task-VALUE",
            'multi-task VALUE',
            datasets_tag="SALT-NLP/mnli_VALUE", #! Problematic
            private=True,
            overwrite_adapter_card=True
        )

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "text-classification"}
    kwargs["language"] = "en"
    kwargs["dataset"] = "multi-task VALUE"

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

