text = """
    # Pancreatitis
    - Lipase: 535 -> 154 -> 145
    - Managed with NBM, IV fluids
    - CT AP and abdo USS: normal
    - Likely secondary to Azathioprine - ceased, never to be used again.
    - Resolved with conservative measures 
    """

import argparse
from ADR_NLP import metrics, data, visualize_ner
from transformers import (
    AutoTokenizer, PreTrainedTokenizerFast, DataCollatorForTokenClassification,
    AutoModelForTokenClassification, Trainer, TrainingArguments)

parser = argparse.ArgumentParser(description='Train an NLP model')
parser.add_argument('--model', type=str, default='austin/deberta-pretrained-large', help='Choose a model from the HF hub')

parser.add_argument('--hub_on', dest='push_to_hub', action='store_true', help='Push the model to the HF hub')
parser.add_argument('--hub_off', dest='push_to_hub', action='store_false', help='Push the model to the HF hub')
parser.set_defaults(push_to_hub=False)

parser.add_argument('--data', type=str, default='annotations.jsonl', help='Path to the data file')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
parser.add_argument('--batch_size', type=int, default=3, help='Batch size')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
parser.add_argument('--seed', type=int, default=42, help='Random seed')

parser.add_argument('--wandb_on', dest='wandb', action='store_true', help='Use wandb for logging')
parser.add_argument('--wandb_off', dest='wandb', action='store_false', help='No wandb logging')
parser.set_defaults(wandb=False)

parser.add_argument('--run_name', type=str, default='adr_nlp', help='Name of the run')

parser.add_argument('--data_file', default='annotated_data')
parser.add_argument('--save_data_dir', default=None, help='Save processed dataset to this directory')
parser.add_argument('--text_col', default='text')
parser.add_argument(
    '--tokenizer_name', 
    default='austin/deberta-pretrained-large',
    help='Name of the tokenizer. Must be a fast tokenizer.')
parser.add_argument('--folds', default=5, type=int, help='Number of folds to split data into.')
args = parser.parse_args()

# Weights & Biases for experiment tracking
if args.wandb:
    import wandb
    wandb.login()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, add_prefix_space=True)
assert isinstance(tokenizer, PreTrainedTokenizerFast)
data_collator = DataCollatorForTokenClassification(tokenizer)

dataset = data.NERdataset(
    args.data_file, 
    args.text_col, 
    tokenizer, 
    args.folds, 
    args.seed,
    args.save_data_dir)

labels = list(dataset.label2id.keys())
cm = metrics.CompMetrics(labels)


train_args = TrainingArguments(
        f"{args.run_name}-finetuned",
        evaluation_strategy = "epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        push_to_hub=False,
        seed=args.seed,
    )

if args.folds > 1:
    for fold, data in dataset.dset.items():
        model = AutoModelForTokenClassification.from_pretrained(
            args.model, 
            label2id = dataset.label2id,
            id2label = dataset.id2label,
            num_labels = len(dataset.label2id))

        train_args.run_name = f'{args.run_name}-{fold}'

        trainer = Trainer(
            model,
            train_args,
            train_dataset=data["train"],
            eval_dataset=data["test"],
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=cm.compute_metrics
        )

        trainer.train()

        doc_level_metrics = metrics.doc_level_metrics(
            trainer,
            data["test"], 
            label_list = labels,
            metric_labels = ['ADR'],
            wandb_log=args.wandb)

        pred_html = visualize_ner.generate_highlighted_text(model, tokenizer, text)
    
        if args.wandb:
            wandb.log({"NER": wandb.Html(pred_html)})

        if args.wandb:
            wandb.finish()

        print(doc_level_metrics)

else:
    model = AutoModelForTokenClassification.from_pretrained(
            args.model, 
            label2id = dataset.label2id,
            id2label = dataset.id2label,
            num_labels = len(dataset.label2id))

    # train_args.push_to_hub = args.push_to_hub
    train_args.run_name = args.run_name

    trainer = Trainer(
        model,
        train_args,
        train_dataset=dataset.dset["train"],
        eval_dataset=dataset.dset["test"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=cm.compute_metrics
        )

    trainer.train()

    doc_level_metrics = metrics.doc_level_metrics(
            trainer,
            dataset.dset["test"], 
            label_list = labels,
            metric_labels = ['ADR'],
            wandb_log=args.wandb)

    pred_html = visualize_ner.generate_highlighted_text(model, tokenizer, text)
    
    if args.wandb:
        wandb.log({"NER": wandb.Html(pred_html)})

    if args.wandb:
        wandb.finish()

    print(doc_level_metrics)
