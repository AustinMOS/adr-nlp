# Natural Language Processing for Adverse Drug Reaction (ADR) Detection

This repo contians code from a project to identify ADRs in discharge summaries at Austin Health. The model uses the HuggingFace Transformers library, beginning with the pretrained deBERTa model. Further MLM pre-training is performed on a large corpus of unannotated discharge summaries. Finally, fine-tuning is peformed on a corpus of annotated discharge summaries. The model performs NER, but final performance is measured at the document level using the maximum token-level score.

We used Weights and Biases for experiment tracking.
