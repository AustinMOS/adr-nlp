from spacy.training import offsets_to_biluo_tags
import spacy


# we will create a function to convert offset formatted labels to BILUO tags
def label_table(dataset):
    nlp = spacy.blank("en")
    tokens = [word['text'] for word in dataset['tokens']]
    dataset['tokens'] = tokens
    dataset['ner_tags'] = offsets_to_biluo_tags(
        nlp(dataset['text']), 
        [(d['start'], d['end'], d['label']) for d in [d for d in (dataset['spans'] or [])]])
    
    return dataset

# this class holds a function for tokenizing and aligning labels
class tokenize_and_align_labels():
    def __init__(self, tokenizer, label_all_tokens=True):
        self.tokenizer = tokenizer
        self.label_all_tokens = label_all_tokens
    def tokenize_align(self, examples):
        tokenized_inputs = self.tokenizer(examples["tokens"], truncation=False, is_split_into_words=True)

        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
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
                    label_ids.append(label[word_idx] if self.label_all_tokens else -100)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs

# function for splitting text into appropriate sized chunks
def split_text(x, max_length=512):
    length = len(x)
    if length > max_length:
        splits = length // max_length
        y = list()
        [y.append(x[i : i + max_length]) for i in range(0, splits*max_length, max_length)]
        if length % max_length > 0:
            y.append(x[splits*max_length : length])
    else:
        y = list([x])
        
    return y

def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))