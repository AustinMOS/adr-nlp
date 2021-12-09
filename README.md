# Natural Language Processing for Adverse Drug Reaction (ADR) Detection

This repo contains code from a project to identify ADRs in discharge summaries at Austin Health. The model uses the HuggingFace Transformers library, beginning with the pretrained DeBERTa model. Further MLM pre-training is performed on a large corpus of unannotated discharge summaries. Finally, fine-tuning is peformed on a corpus of annotated discharge summaries (annotated using [Prodigy](https://prodi.gy)). The model performs NER, but final performance is measured at the document level using the maximum token-level score.

We used [Weights and Biases](https://wandb.ai) for experiment tracking.

The *pretrain* script takes a folder containing discharge summaries stored in CSV folders, tokenizes and continues MLM training on [deberta-base](https://huggingface.co/microsoft/deberta-base).

Fine-tuning can then be performed with the *finetune* script using CLI commands. This script assumes the data is either a JSONL file of annotated text exported from Prodigy (`--datafile example.jsonl`), or a saved HuggingFace Datasets. If you run this script once on a JSONL file of annotations, you can choose to save the Dataset into a folder (`--save_data_dir "save_to_here"`) and use this for subsequent training runs (`--datafile "save_to_here"`).

Example usage:
```bash
python .\finetune.py --folds 5 --epochs 15 --lr 5e-5 --wandb_on --hub_off --project 'CLI Tests' --run_name cross-validation --datafile 'data'
```