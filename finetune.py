import argparse
from ADR_NLP.utils import none_or_str
from ADR_NLP.data import NERdataset
from ADR_NLP.metrics import CompMetrics, doc_level_metrics
from ADR_NLP.visualize_ner import generate_highlighted_text
from transformers import (
    AutoTokenizer, PreTrainedTokenizerFast, DataCollatorForTokenClassification,
    AutoModelForTokenClassification, Trainer, TrainingArguments)
from pathlib import Path


def main(
    model_name, push_to_hub, datafile, epochs, batch_size, lr, 
    weight_decay, seed, wb, run_name, project, text_file, 
    save_data_dir, text_col, tokenizer_name, folds
    ):

    if wb:
        import wandb
        wandb.init(project=project)
        text = Path(text_file).read_text()

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, add_prefix_space=True)
    assert isinstance(tokenizer, PreTrainedTokenizerFast)
    data_collator = DataCollatorForTokenClassification(tokenizer)

    dataset = NERdataset(
        datafile, 
        text_col, 
        tokenizer, 
        folds, 
        seed,
        save_data_dir)

    labels = list(dataset.label2id.keys())
    cm = CompMetrics(labels)


    train_args = TrainingArguments(
            f"{run_name}-finetuned",
            evaluation_strategy = "epoch",
            learning_rate=lr,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=weight_decay,
            push_to_hub=False,
            seed=seed,
        )

    if folds > 1:
        for fold, data in dataset.dset.items():
            model = AutoModelForTokenClassification.from_pretrained(
                model_name, 
                label2id = dataset.label2id,
                id2label = dataset.id2label,
                num_labels = len(dataset.label2id))

            train_args.run_name = f'{run_name}-{fold}'

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

            doc_metrics = doc_level_metrics(
                trainer,
                data["test"], 
                label_list = labels,
                metric_labels = ['ADR'],
                wandb_log=wb)

            pred_html = generate_highlighted_text(model, tokenizer, text)
        
            if wb:
                wandb.log({"NER": wandb.Html(pred_html)})

            if wb:
                wandb.finish()

            print(doc_metrics)

    else:
        model = AutoModelForTokenClassification.from_pretrained(
                model_name, 
                label2id = dataset.label2id,
                id2label = dataset.id2label,
                num_labels = len(dataset.label2id))

        train_args.push_to_hub = push_to_hub
        train_args.run_name = run_name

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

        doc_metrics = doc_level_metrics(
                trainer,
                dataset.dset["test"], 
                label_list = labels,
                metric_labels = ['ADR'],
                wandb_log=wb)

        pred_html = generate_highlighted_text(model, tokenizer, text)
        
        if wb:
            wandb.log({"NER": wandb.Html(pred_html)})

        if wb:
            wandb.finish()

        print(doc_metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an NLP model')
    parser.add_argument('--model_name', type=str, default='austin/deberta-pretrained-large', help='Choose a model from the HF hub')

    parser.add_argument('--hub_on', dest='push_to_hub', action='store_true', help='Push the model to the HF hub')
    parser.add_argument('--hub_off', dest='push_to_hub', action='store_false', help='Push the model to the HF hub')
    parser.set_defaults(push_to_hub=False)

    parser.add_argument('--datafile', type=str, default='annotations.jsonl', help='Path to the data file')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=3, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    parser.add_argument('--wandb_on', dest='wb', action='store_true', help='Use wandb for logging')
    parser.add_argument('--wandb_off', dest='wb', action='store_false', help='No wandb logging')
    parser.set_defaults(wb=False)

    parser.add_argument('--run_name', type=str, default='adr_nlp', help='Name of the run for wandb')
    parser.add_argument('--project', type=str, default='adr_nlp', help='Name of the project for wandb')
    parser.add_argument('--text_file', type=str, default='testing.txt', help='If logging with wandb, a file containing a text to test NER on')

    parser.add_argument('--save_data_dir', type=none_or_str, nargs='?', default=None, help='Save processed dataset to this directory')
    parser.add_argument('--text_col', type=str, default='text', help='Name of column in datafile that contains the text')
    parser.add_argument(
        '--tokenizer_name', 
        default='austin/deberta-pretrained-large',
        help='Name of the tokenizer. Must be a fast tokenizer.')
    parser.add_argument('--folds', default=5, type=int, help='Number of folds to split data into.')
    args = parser.parse_args()

    main(**vars(args))