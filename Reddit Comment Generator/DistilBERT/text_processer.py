from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import numpy as np
import torch


class TextPreprocesser:
    def __init__(self, tokenizer, sentences, labels = None, MAX_LEN = 128, batch_size = 32, train = False):
        self.tokenizer = tokenizer
        self.MAX_LEN = MAX_LEN
        self.batch_size = batch_size
        self.sentences = sentences
        self.labels = labels
        self.train = train

        # Create attention masks
        self.attention_masks = []

    def tokenize(self):
        self.sentences = ["[CLS] " + str(sentence) + " [SEP]" for sentence in tqdm(self.sentences,
                                                                                   total = len(self.sentences),
                                                                                   desc = "Converting Sentences: ")]
        tokenized_texts = [self.tokenizer.tokenize(sent) for sent in tqdm(self.sentences,
                                                                          total = len(self.sentences),
                                                                          desc = "Tokenizing: ")]

        # Use the DistilBERT tokenizer to convert the tokens to their index numbers in the DistilBERT vocabulary
        self.input_ids = [self.tokenizer.convert_tokens_to_ids(x) for x in tqdm(tokenized_texts,
                                                                                total = len(tokenized_texts),
                                                                                desc = "Token to IDs: ")]

        # Pad input tokens
        print("Padding input tokens...")
        self.input_ids = pad_sequences(self.input_ids, maxlen=self.MAX_LEN, dtype="long", truncating="post", padding="post")
        print("Done!")

        # Create a mask of 1s for each token followed by 0s for padding
        for seq in tqdm(self.input_ids, total = len(self.input_ids), desc = "Creating attention masks: "):
            seq_mask = [float(i>0) for i in seq]
            self.attention_masks.append(seq_mask)

    def train_valid_split(self, test_size = 0.2):
        self.train_inputs, self.validation_inputs, self.train_labels, self.validation_labels = train_test_split(self.input_ids,
                                                                                                                self.labels,
                                                                                                                random_state = 2018,
                                                                                                                test_size = test_size)
        self.train_masks, self.validation_masks, _, _ = train_test_split(self.attention_masks,
                                                                         self.input_ids,
                                                                         random_state = 2018,
                                                                         test_size = test_size)

    def convert_to_tensors(self):
        print("Converting to tensors...")
        if self.train:
            self.train_inputs = torch.tensor(self.train_inputs)
            self.validation_inputs = torch.tensor(self.validation_inputs)
            self.train_labels = torch.tensor(self.train_labels)
            self.validation_labels = torch.tensor(self.validation_labels)
            self.train_masks = torch.tensor(self.train_masks)
            self.validation_masks = torch.tensor(self.validation_masks)

            train_data = TensorDataset(self.train_inputs, self.train_masks, self.train_labels)
            train_sampler = RandomSampler(train_data)
            self.train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size)

            validation_data = TensorDataset(self.validation_inputs, self.validation_masks, self.validation_labels)
            validation_sampler = SequentialSampler(validation_data)
            self.validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=self.batch_size)

        else:
            prediction_inputs = torch.tensor(self.input_ids)
            prediction_masks = torch.tensor(self.attention_masks)

            if self.labels is None:
                prediction_data = TensorDataset(prediction_inputs, prediction_masks)
                prediction_sampler = SequentialSampler(prediction_data)
                self.prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=self.batch_size)

            else:
                prediction_labels = torch.tensor(self.labels)
                prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
                prediction_sampler = SequentialSampler(prediction_data)
                self.prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=self.batch_size)
        print("Done!")
