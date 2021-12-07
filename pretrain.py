import datasets
from transformers import (AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments,
                          Trainer, AutoModelForMaskedLM, PreTrainedTokenizerFast)

from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, PreTrainedTokenizerFast
import re
import glob

text_path = "../pretraining_data" # Path to documents to use for pre-training (stored in CSV files)
base_model = "microsoft/deberta-base" # Base pre-trained model
save_to_hub = "organization/model_name" # Directory where the model will be saved in HF hub
report = "wandb" # report to Weights and Biases
run_name = "run_name" # Name of the run for reporting and local storage
other_data_sources =  False # False for this project. Set to True if also using other texts (e.g. radiology reports, pathology reports etc.)


# Function to remove discharge summary "bloat" (e.g. long copy-and-paste sections like medication list and pathology results)
def dcsumm_body(text):
    if text['TEXT']:
        reduced = re.search('PRINCIPAL DIAGNOSIS(.*?)DISCHARGE|PRESCRIBED MEDICATION', text['TEXT'], flags=re.S)
        if reduced:
            text['TEXT'] = reduced.group(1)
        else:
            text['TEXT'] = ''
    else:
        text['TEXT'] = ''
    return text


def tokenize_function(examples, tokenizer):
    return tokenizer(examples["TEXT"])


def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    total_length = (total_length // 512) * 512
    result = {
        k: [t[i: i + 512] for i in range(0, total_length, 512)]
        for k, t in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


datafiles = glob.glob(f"{text_path}/*")
tokenizer = AutoTokenizer.from_pretrained(base_model)
assert isinstance(tokenizer, PreTrainedTokenizerFast)

dcsumm_datafiles=[datafile for datafile in datafiles if 'summaries' in datafile]
other_datafiles=[datafile for datafile in datafiles if 'summaries' not in datafile]

dataset = load_dataset("csv", data_files=dcsumm_datafiles) \
    .map(dcsumm_body) \
    .filter(lambda example: (example['TEXT'] is not None) & (example['TEXT'] != '')) \
    .remove_columns(['EPISODE_ID', 'PATIENT_ID', 'DOC_ID', 'START_DTTM'])

if other_data_sources:
    other_dataset = load_dataset("csv", data_files=other_datafiles) \
        .filter(lambda example: (example['TEXT'] is not None) & (example['TEXT'] != ''))

    dataset = concatenate_datasets([dataset['train'], other_dataset['train']])

tokenised = dataset.map(tokenize_function, fn_kwargs={'tokenizer': tokenizer},
                        batched=True, batch_size=5000,
                        remove_columns=["TEXT"])

tokenised = tokenised.map(
    group_texts,
    batched=True,
    num_proc=10
)

tokenised.save_to_disk("tokenized-texts")

model = AutoModelForMaskedLM.from_pretrained(base_model)
tokenizer = AutoTokenizer.from_pretrained(base_model)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
assert isinstance(tokenizer, PreTrainedTokenizerFast)

train_test = datasets.load_from_disk('tokenized-texts')
train_test = train_test.train_test_split(test_size=0.05)

training_args = TrainingArguments(
    run_name,
    evaluation_strategy = "steps",
    eval_steps = 40_000,
    learning_rate=5e-5,
    weight_decay=0.01,
    per_device_train_batch_size=3,
    per_device_eval_batch_size=3,
    report_to=report,
    run_name=run_name,
    num_train_epochs=5,
    save_steps=40_000,
    hub_model_id=save_to_hub,
    push_to_hub=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_test["train"],
    eval_dataset=train_test["test"],
    data_collator=data_collator,
)

trainer.train()
trainer.push_to_hub()