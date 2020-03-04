import torch
import numpy as np
from tqdm import tqdm


class ModelPredictor:

    def __init__(self, model, device, prediction_dataloader, plotlabels=None, labels_available=False):
        self.model = model
        self.predictions = []
        self.true_labels = []
        self.device = device
        self.labels_available = labels_available
        self.prediction_dataloader = prediction_dataloader
        self.plotlabels = plotlabels

    def predict(self):
        self.model.eval()
        n = len(self.prediction_dataloader)
        for batch in tqdm(self.prediction_dataloader, total=len(self.prediction_dataloader), desc="Evaluating: "):
            # Add batch to GPU
            batch = tuple(t.to(self.device) for t in batch)

            # Unpack the inputs from our dataloader
            if self.labels_available:
                b_input_ids, b_input_mask, b_labels = batch
            else:
                b_input_ids, b_input_mask = batch

            # Telling the model not to compute or store gradients, saving memory and speeding up prediction
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                outputs = self.model(b_input_ids, attention_mask=b_input_mask)
                logits = outputs[0]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            if self.labels_available:
                label_ids = b_labels.to('cpu').numpy()

            # Store predictions
            self.predictions.append(logits)
            if self.labels_available:
                self.true_labels.append(label_ids)

        self._retrieveClasses()
        if self.labels_available:
            self._printStats()

    def _retrieveClasses(self):
        self.flat_predictions = [item for sublist in self.predictions for item in sublist]
        self.flat_predictions = np.argmax(self.flat_predictions, axis=1).flatten()
        if self.labels_available:
            self.flat_true_labels = [item for sublist in self.true_labels for item in sublist]

    def _printStats(self, sep='-', sep_len=40):
        print('Accuracy = %.3f' % accuracy_score(self.flat_true_labels, self.flat_predictions))
        print(sep*sep_len)
        print('Classification report:')
        print(classification_report(self.flat_true_labels, self.flat_predictions))
        print(sep*sep_len)
        print('Confusion matrix')
        cm=confusion_matrix(self.flat_true_labels, self.flat_predictions)
        cm = cm / np.sum(cm, axis=1)[:,None]
        heatmap = sns.heatmap(cm,
            xticklabels=self.plotlabels,
            yticklabels=self.plotlabels,
            annot=True, cmap = 'YlGnBu')
        fig = heatmap.get_figure()
        if self.labels_available:
            fig.savefig("heatmap_holdout.png")
        else:
            fig.savefig("new_predictions.png")
