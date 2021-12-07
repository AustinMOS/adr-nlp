"""
This script is used to create our final tokenized data set
using the HuggingFace Datasets library.

The helper functions from data_helpers.py are used to process annotated
text, generated using the Prodigy NLP tool.

The final dataset will be chunked according to the maximum tokens allowed
in our model (512)
"""
import datasets
from datasets import Features, Sequence
from datasets.features import Value, ClassLabel
from ADR_NLP.data_helpers import (
    split_text, tokenize_and_align_labels, 
    label_table, split)
import random

# A helper function that returns True if 'jsonl' is found in a string, otherwise False
def is_jsonl(string):
    if 'jsonl' in string:
        return True
    else:
        return False

"""
A class to hold the data and labels for the model.
Can be initialized from a JSONL file and procesed, or from a preprocessed Dataset that has been saved.
"""
class NERdataset():
    def __init__(self, data_file, text_col, tokenizer, folds, seed, save = None):
        self.tokenizer = tokenizer
        self.text_col = text_col
        self.folds = folds
        self.seed = seed
        self.data_file = data_file
        self.save = save
        self.ta = tokenize_and_align_labels(tokenizer)

    # Load dataset
        self.load()
    # Process the data (and save if save is not None)
        self.process()

    # Load using datasets.Dataset.from_json if the data is an unprocessed jsonl file, otherwise load from a preprocessed dataset using load_dataset
    def load(self):
        if is_jsonl(self.data_file):
            self.dataset = datasets.Dataset.from_json(self.data_file)
        else:
            self.dataset = datasets.load_from_disk(self.data_file)
        return self

    def process_jsonl(self):
        rm_cols = list(set(self.dataset.column_names) - set([self.text_col, 'tokens','ner_tags']))
        self.dataset = self.dataset.map(label_table, remove_columns=rm_cols)

        ner_names = list(set([it for sl in self.dataset['ner_tags'] for it in sl]))
        features = Features(
            {self.text_col: Value(dtype='string', id=None),
            'tokens': Sequence(feature=Value(dtype='string', id=None), length=-1, id=None),
            'ner_tags': Sequence(feature=ClassLabel(names=ner_names), length=-1, id=None)}
            )
        self.dataset = self.dataset.map(features.encode_example, features=features)
        if self.save:
            self.dataset.save_to_disk(self.save)

    def process_dataset(self):
        label_list = self.dataset.features['ner_tags'].feature.names
        ids = range(len(label_list))
        self.label2id = dict(zip(label_list, ids))
        self.id2label = dict(zip(ids, label_list))

        tokenized_dataset = self.dataset.map(self.ta.tokenize_align, batched=True)

        split_dataset = datasets.Dataset.from_pandas(
            tokenized_dataset
            .remove_columns(['ner_tags', 'tokens', self.text_col])
            .to_pandas()
            .applymap(split_text, max_length=self.tokenizer.model_max_length)
            .explode(['attention_mask', 'input_ids', 'labels', 'token_type_ids'])
            .reset_index())

        if self.folds > 1:

            indices = list(dict.fromkeys(split_dataset['index']))

            folds_list = list(split(indices, n=self.folds))
            random.shuffle(indices)

            self.dset = dict()
            for i in range(self.folds):
                test = [indices[index] for index in folds_list[i]]
                train = list(set(indices) - set(test))
                self.dset[f'fold{i}'] = datasets.DatasetDict({
                'train': split_dataset.filter(lambda example: example['index'] in train),
                'test': split_dataset.filter(lambda example: example['index'] in test)
                })
        
        else:
            self.dset = split_dataset.train_test_split(test_size=0.2, seed=self.seed)

    # Process the data
    def process(self):
        if is_jsonl(self.data_file):
            self.process_jsonl()
            self.process_dataset()
        else:
            self.process_dataset()
            

    

    def __repr__(self):
        return f'Data(data_file={self.data_file}, text_col={self.text_col}, tokenizer={self.tokenizer}, folds={self.folds}, seed={self.seed})'
    