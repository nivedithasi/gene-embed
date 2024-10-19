'''Evaluate method'''
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.special import softmax
from utils_stanford import (precision_at_top_K, enrichment_at_top_K)

def get_scores(y_actual, y_prob):
    '''Compute scores'''
    auroc = roc_auc_score(y_actual, y_prob)
    auprc = average_precision_score(y_actual, y_prob)
    num_positive = sum(y_actual)  # number of positive pairs in test set
    precision = precision_at_top_K(y_actual, y_prob, num_positive)
    enrichment = enrichment_at_top_K(y_actual, y_prob, num_positive)

    return {'auroc': auroc, 'auprc': auprc, 'p@top{k}': precision,
            'enrichment@top{k}': enrichment}

def predict(dataloader, model, loss_fn, device):
    '''get predictions for a dataset'''
    model.eval()
    test_loss = 0
    y_prob = []
    y_actual = []
    meta = []

    num_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            X, y = batch[0].to(device), batch[1].to(device)
            if len(batch) == 3:
                # There is metadata
                names = dataloader.dataset.meta_cols
                meta_batch = pd.DataFrame(index=names, data=batch[-1]).T
                meta.append(meta_batch)

            logits = model(X)
            test_loss += loss_fn(logits, y).item()
            # logits, need to softmax to get probabilities
            logits = logits.to('cpu').numpy()
            probs = softmax(logits, axis=1)
            y_prob.extend(probs[:, 1])  # predicted probability
            y_actual.extend(y.to('cpu').numpy())

            num_batches += 1

    test_loss /= num_batches

    y_prob, y_actual = np.array(y_prob), np.array(y_actual)
    predictions = pd.concat(meta)
    predictions['y_prob'] = y_prob
    predictions['y_actual'] = y_actual
    return predictions, test_loss

class ExperimentResult():
    '''Wrapper for prediction and metrics'''

    def __init__(self):
        self.scores = None
        self.predictions = None

    def compute(self, model, loss_fn, loader, device):
        '''Compute performance of a trained model in a given dataset'''
        predictions, loss = predict(loader, model, loss_fn, device)
        self.scores = get_scores(predictions['y_actual'].values,
                                 predictions['y_prob'].values)
        self.predictions = predictions
        self.scores['loss'] = loss
