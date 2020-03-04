from transformers import AdamW
from tqdm import tqdm
import numpy as np


class ModelTrainer:
    def __init__(self, model, tokenizer, train_dataloader, validation_dataloader, epochs=2):
        self.model = model
        self.train_loss_set = []
        self.epochs = epochs
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.tokenizer = tokenizer

    def set_optimizer(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]

        self.optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5)
        self.loss_func = CrossEntropyLoss()

    def _flatAccuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)

    def train(self):
        for epoch in tqdm(range(self.epochs), desc="Epoch: "):
            # Set our model to training mode (as opposed to evaluation mode)
            self.model.train()

            # Tracking variables
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0

            # Train the data for one epoch
            with tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader), desc="Training: ") as t:
                for step, batch in t:
                    # Add batch to GPU
                    batch = tuple(t.to(device) for t in batch)
                    # Unpack the inputs from our dataloader
                    b_input_ids, b_input_mask, b_labels = batch

                    # Clear out the gradients (by default they accumulate)
                    self.optimizer.zero_grad()

                    # Forward pass
                    outputs = self.model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                    loss = outputs[0]
                    logits = outputs[1]
                    self.train_loss_set.append(loss.item())

                    # Backward pass
                    loss.backward()

                    # Update parameters and take a step using the computed gradient
                    self.optimizer.step()

    #                 if step % 100 == 0:
    #                     print("Epoch: {}, Step: {}, Loss: {}".format(epoch, step, loss.item()))

                # Update tracking variables
                tr_loss += loss.item()
                nb_tr_examples += b_input_ids.size(0)
                nb_tr_steps += 1

                t.set_postfix(train_loss=(tr_loss/nb_tr_steps))

                # Validation
                self._evaluator(epoch)

    def _evaluator(self, epoch):
        # Put model in evaluation mode to evaluate loss on the validation set
        model.eval()

        # Tracking variables
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # Evaluate data for one epoch
        with tqdm(self.validation_dataloader, total=len(self.validation_dataloader), desc="Validating: ") as t:
            for batch in t:
                # Add batch to GPU
                batch = tuple(t.to(device) for t in batch)
                # Unpack the inputs from our dataloader
                b_input_ids, b_input_mask, b_labels = batch
                # Telling the model not to compute or store gradients, saving memory and speeding up validation
                with torch.no_grad():
                    # Forward pass, calculate logit predictions
                    output = self.model(b_input_ids, attention_mask=b_input_mask)
                    logits = output[0]

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            tmp_eval_accuracy = self._flatAccuracy(logits, label_ids)

            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

            t.set_postfix(val_acc=(eval_accuracy/nb_eval_steps))

            directory = "distilbert_epoch_" + str(epoch + 3) + '_wholedata/'

            if not os.path.exists(directory):
                os.makedirs(directory)

            self.model.save_pretrained(directory)  # save
            self.tokenizer.save_pretrained(directory)  # save
