import numpy as np
from datasets import load_metric
from scipy.special import softmax
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import pandas as pd
import re
import wandb

class CompMetrics():
    def __init__(self, label_list):
        self.label_list = label_list
        self.metric = load_metric("seqeval")
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


def doc_level_metrics(trainer, dataset, label_list, metric_labels = ['ADR'], wandb_log = False):
    probs, _, _ = trainer.predict(dataset)
    doc_metrics = dict()
    for label in metric_labels:
        label_loc = [i for i, item in enumerate(label_list) if re.search(label, item)]
        probs_soft = softmax(probs, axis=2)[:,:,label_loc].max(axis=(1,2))

        doc_level = dataset.map(
            lambda example: 
            {'Truth': any(item in example['labels'] for item in label_loc)}, 
            remove_columns = ['attention_mask', 'input_ids', 'labels', 'token_type_ids']).to_pandas()
        doc_level['Prob'] = probs_soft

        truth = doc_level.groupby('index')['Truth'].max().values
        estimate = doc_level.groupby('index')['Prob'].max().values


        pr = average_precision_score(truth, estimate)
        auc = roc_auc_score(truth, estimate)
        fpr, tpr, threshold = roc_curve(truth, estimate)
        roc_table = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'threshold': threshold})
        doc_metrics[label] = {"AUC": auc, "PR": pr}
        if wandb_log:
            wandb.log({label + " AUC": auc, label + " PR": pr})
            wandb.log({"roc" : wandb.Table(dataframe=roc_table)})

    return doc_metrics