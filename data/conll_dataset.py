import torch
from torch.utils import data
from torch.utils.data import Dataset
from datasets.arrow_dataset import Dataset as HFDataset
from datasets.load import load_dataset, load_metric
from transformers import AutoTokenizer, DataCollatorForTokenClassification, AutoConfig
import numpy as np

# pos_tags: a list of classification labels, with possible values including " (0), '' (1), # (2), $ (3), ( (4).
# chunk_tags: a list of classification labels, with possible values including O (0), B-ADJP (1), I-ADJP (2), B-ADVP (3), I-ADVP (4).
# ner_tags: a list of classification labels, with possible values including O (0), B-PER (1), I-PER (2), B-ORG (3), I-ORG (4) B-LOC (5), I-LOC (6) B-MISC (7), I-MISC (8).

POS_LIST = [ '"', "''", '#', '$', '(']
CHUNK_LIST = ['O', 'B-ADJP', 'I-ADJP', 'B-ADVP', 'I-ADVP']
NER_LIST = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
CONLL05_LIST = ['B-C-AM-TMP', 'B-C-AM-DIR', 'B-C-A2', 'B-R-AM-EXT', 'B-C-A0', 'I-AM-NEG', 'I-AM-ADV', 'B-C-V', 'B-C-AM-MNR', 'B-R-A3', 'I-AM-TM', 'B-V', 'B-R-A4', 'B-A5', 'I-A4', 'I-R-AM-LOC', 'I-C-A1', 'B-R-AA', 'I-C-A0', 'B-C-AM-EXT', 'I-C-AM-DIS', 'I-C-A5', 'B-A0', 'B-C-A4', 'B-C-AM-CAU', 'B-C-AM-NEG', 'B-AM-NEG', 'I-AM-MNR', 'I-R-A2', 'I-R-AM-TMP', 'B-AM', 'I-R-AM-PNC', 'B-AM-LOC', 'B-AM-REC', 'B-A2', 'I-AM-EXT', 'I-V', 'B-A3', 'B-A4', 'B-R-A0', 'I-AM-MOD', 'I-C-AM-CAU', 'B-R-AM-CAU', 'B-A1', 'B-R-AM-TMP', 'I-R-AM-EXT', 'B-C-AM-ADV', 'B-AM-ADV', 'B-R-A2', 'B-AM-CAU', 'B-R-AM-DIR', 'I-A5', 'B-C-AM-DIS', 'I-C-AM-MNR', 'B-AM-PNC', 'I-C-AM-LOC', 'I-R-A3', 'I-R-AM-ADV', 'I-A0', 'B-AM-EXT', 'B-R-AM-PNC', 'I-AM-DIS', 'I-AM-REC', 'B-C-AM-LOC', 'B-R-AM-ADV', 'I-AM', 'I-AM-CAU', 'I-AM-TMP', 'I-A1', 'I-C-A4', 'B-R-AM-LOC', 'I-C-A2', 'B-C-A5', 'O', 'B-R-AM-MNR', 'I-C-A3', 'I-R-AM-DIR', 'I-AM-PRD', 'B-AM-TM', 'I-A2', 'I-AA', 'I-AM-LOC', 'I-AM-PNC', 'B-AM-MOD', 'B-AM-DIR', 'B-R-A1', 'B-AM-TMP', 'B-AM-MNR', 'I-R-A0', 'B-AM-PRD', 'I-AM-DIR', 'B-AM-DIS', 'I-C-AM-ADV', 'I-R-A1', 'B-C-A3', 'I-R-AM-MNR', 'I-R-A4', 'I-C-AM-PNC', 'I-C-AM-TMP', 'I-C-V', 'I-A3', 'I-C-AM-EXT', 'B-C-A1', 'B-AA', 'I-C-AM-DIR', 'B-C-AM-PNC']

MAP_DICT = {
    'ner': NER_LIST,
    'chunk': CHUNK_LIST,
    'pos': POS_LIST,
    'conll05':CONLL05_LIST
}

class CoNLL(Dataset):
    def __init__(self, model_name:str, aps: bool) -> None:
        super().__init__()
        # self.task = task + '_tags'
        self.input, self.labels, self.label_mask, self.attention_mask = [], [], [], []
        self.ignore_columns = ['tags','id','tokens', 'index']

        data = load_dataset('data/load_dataset.py')
        
        use_fast = False if 'v2' in model_name else True
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            use_fast=use_fast,
            revision='main',
            add_prefix_space=aps,
        )

        self.label_list = CONLL05_LIST
        self.label_to_id = {l: i for i, l in enumerate(self.label_list)}
        num_labels = len(self.label_list)

        self.train_data = data['train'].map(
            self.tokenize_and_align_labels,
            batched=True,
            load_from_cache_file=True,
            desc="Running tokenizer on train dataset",
        )
        self.dev_data = data['validation'].map(
            self.tokenize_and_align_labels,
            batched=True,
            load_from_cache_file=True,
            desc="Running tokenizer on validation dataset",
        )
        self.test_wsj = data['test_wsj'].map(
            self.tokenize_and_align_labels,
            batched=True,
            load_from_cache_file=True,
            desc="Running tokenizer on WSJ test dataset",
        )

        self.test_brown = data['test_brown'].map(
            self.tokenize_and_align_labels,
            batched=True,
            load_from_cache_file=True,
            desc="Running tokenizer on Brown test dataset",
        )

        self.data_collator = DataCollatorForTokenClassification(self.tokenizer, pad_to_multiple_of=8)
        self.metric = load_metric('data/metric.py')
        self.config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
            # label2id=self.label_to_id,
            # id2label={i: l for l, i in self.label_to_id.items()},
            # finetuning_task=task,
            revision='main',
        )

    def compute_metrics(self, p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        true_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = self.metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

    def tokenize_and_align_labels(self, examples):        
        for i, tokens in enumerate(examples['tokens']):
            examples['tokens'][i] = tokens + ["[SEP]"] + [tokens[int(examples['index'][i])]]

        tokenized_inputs = self.tokenizer(
            examples['tokens'],
            padding=False,
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )

        # print(tokenized_inputs['input_ids'][0])

        labels = []
        for i, label in enumerate(examples['tags']):
            word_ids = [None]
            for j, word in enumerate(examples['tokens'][i][:-2]):
                token = self.tokenizer.encode(word, add_special_tokens=False)
                word_ids += [j] * len(token)
            word_ids += [None]
            verb = examples['tokens'][i][int(examples['index'][i])]
            word_ids += [None] * len(self.tokenizer.encode(verb, add_special_tokens=False))
            word_ids += [None]
            # print(word_ids)
            # exit()
            
            # word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            # print(len(label))
            # print(word_ids)
            # exit()
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    def get_label_list(self, labels):
        unique_labels = set()
        for label in labels:
            unique_labels = unique_labels | set(label)
        label_list = list(unique_labels)
        label_list.sort()
        return label_list
