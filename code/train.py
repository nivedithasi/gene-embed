'''Functions to train a model'''
from pathlib import Path
import torch
import numpy as np
from tqdm.auto import tqdm
from scipy.special import softmax
from evaluate import ExperimentResult, get_scores

class Trainer():
    '''Object to wrap training loop'''

    def __init__(self, model, loss_fn, optimizer, device):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.device = device

    def train_loop(self, dataloader):
        '''Gradient descent loop'''
        self.model.train()
        y_prob = []
        y_actual = []
        total_loss = 0
        num_batches = 0
        for X, y in dataloader:
            X, y = X.to(self.device), y.to(self.device)
            
            # Compute prediction and loss
            logits = self.model(X)
            loss = self.loss_fn(logits, y)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            num_batches += 1

            # get predictions
            logits = logits.detach().cpu().numpy()
            probs = softmax(logits, axis=1)
            y_prob.extend(probs[:, 1])  # predicted probability
            y_actual.extend(y.cpu().numpy())

        total_loss /= num_batches
        return total_loss, np.asarray(y_actual), np.asarray(y_prob)

    def main_loop(self, train_loader, val_loader, epochs, criteria, model_path, stop_cutoff, no_save=False):
        '''Run the training loop for several epochs'''
        
        if not no_save:
            Path(model_path).parent.mkdir(parents=True, exist_ok=True)

        learning_log = []
        max_scores = {criteria: float('-inf')}
        for i in tqdm(range(epochs), leave=False):

            train_loss, y_actual, y_prob = self.train_loop(train_loader)
            train_scores = get_scores(y_actual, y_prob)
            train_scores['loss'] = train_loss
            train_scores['subset'] = 'train'
            train_scores['epoch'] = i

            val_result = ExperimentResult()
            val_result.compute(self.model, self.loss_fn,
                               val_loader, self.device)
            val_scores = val_result.scores
            val_scores['subset'] = 'valid'
            val_scores['epoch'] = i

            learning_log.extend([train_scores, val_scores])

            if val_scores[criteria] > max_scores[criteria]:
                max_scores = val_scores
                epochs_since_max = 0
                if not no_save:
                    torch.save(self.model.state_dict(), model_path)
            else:
                epochs_since_max += 1
                if epochs_since_max > stop_cutoff:
                    print("Early stopping")
                    break
        return learning_log
