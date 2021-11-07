from functools import partial

import numpy as np
import sklearn.metrics as sklearn_metrics

binary_metrics = {
    "accuracy": sklearn_metrics.accuracy_score,
    "precision": sklearn_metrics.precision_score,
    "recall": sklearn_metrics.recall_score,
    "f1": sklearn_metrics.f1_score,
    "matthews_corrcoef": sklearn_metrics.matthews_corrcoef,
    "roc_auc": sklearn_metrics.roc_auc_score,
}


metrics = {
    "accuracy": sklearn_metrics.accuracy_score,
    "f1-macro": partial(sklearn_metrics.f1_score, average="macro"),
}


def eval_cls(results, **kwargs):
    predictions = np.array([result["prediction"] for result in results])
    labels = np.array([result["label"] for result in results])
    is_binary = len(set(labels.tolist())) < 3
    results = {
        metric: round(f(labels, predictions) * 100, 2)
        for metric, f in (binary_metrics.items() if is_binary else metrics.items())
    }
    return {
        "results": results,
        "best_score": results["f1" if is_binary else "f1-macro"],
    }
