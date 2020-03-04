import torch
from transformers import DistilBertTokenizer, DistilBertConfig, DistilBertForSequenceClassification
from .dataset_loader import DatasetLoader
from .model_predictor import ModelPredictor
from .text_processer import TextPreprocesser
from .model_trainer import ModelTrainer


class DistilBERTModel:
    LABELS = {0: 'funny',
              1: 'soccer',
              2: 'teenagers',
              3: 'learnmachinelearning',
              4: 'gameofthrones',
              5: 'Showerthoughts',
              6: 'unpopularopinion',
              7: 'politics',
              8: 'worldnews'}

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, data, holdout = 0.01):

        if not self.model:
            self.load_pretrained_model()

        # Load the dataset
        dataset_loader = DatasetLoader(data = data, holdout = holdout)
        dataset_loader.transform()
        dataset_loader.prepare_holdout()

        # Preprocess the data
        preprocesser = TextPreprocesser(tokenizer = self.tokenizer,
                                        sentences = dataset_loader.sentences,
                                        labels = dataset_loader.labels,
                                        train = True)
        preprocesser.tokenize()
        preprocesser.train_valid_split()
        preprocesser.convert_to_tensors()

        # Train the model
        trainer = ModelTrainer(model = self.model,
                               tokenizer = self.tokenizer,
                               train_dataloader = preprocesser.train_dataloader,
                               validation_dataloader = preprocesser.validation_dataloader)
        trainer.set_optimizer()
        trainer.train()
        torch.cuda.empty_cache()

    def predict(self, data):
        preprocesser = TextPreprocesser(sentences = data, tokenizer = self.tokenizer)
        preprocesser.tokenize()
        preprocesser.convert_to_tensors()

        predictor = ModelPredictor(self.model, self.device, prediction_dataloader = preprocesser.prediction_dataloader)
        predictor.predict()
        prediction = predictor.flat_predictions[0]
        return self.LABELS[prediction]

    def load_pretrained_model(self, input_path = './DistilBERT/pre_trained_model/'):
        self.model = DistilBertForSequenceClassification.from_pretrained(input_path, num_labels = 9)
        self.tokenizer = DistilBertTokenizer.from_pretrained(input_path, do_lower_case = True)
