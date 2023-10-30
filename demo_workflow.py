from transformers import RobertaPreTrainedModel, RobertaModel
import torch
from torch import nn


class BertForMultiLabelClassification(RobertaPreTrainedModel):
    def __init__(self, config):
        super(BertForMultiLabelClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = RobertaModel.from_pretrained("pdelobelle/robbert-v2-dutch-base")
        self.dropout = nn.Dropout(0.1) #nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        logits = self.sigmoid(logits)
        return logits
    

class DemoFunctions:

    def __init__(self, model, tokenizer):
        
        self.model = model
        self.tokenizer = tokenizer

        self.id2label_parliament = {
        0: "MC8: Economic Impact & Recovery",
        1: "MC13: Government Policy & Response",
        2: "MC15: Labor & Employment Dynamics",
        3: "MC16: Legal & Human Rights",
        4: "Other",
        5: "MC21: Public Health & Preventative Measures"
        }   
        
        self.id2label_twitter = {
            0: "MC10: Epidemiology & Virus Transmission",
            1: "MC13: Government Policy & Response",
            2: "MC17: Media, Public Perception & Culture",
            3: "Other",
            4: "MC21: Public Health & Preventative Measures",
            5: "MC24: Social Relations & Community Dynamics"
        }

    def preprocess(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding="max_length")
        return inputs

    # Step 4: Model Prediction
    def predict(self, text):
        inputs = self.preprocess(text)
        with torch.no_grad():
            probabilities = self.model(**inputs)
        return probabilities

    # Step 5: Decoding the Predictions
    def decode_predictions_parliament(self, probabilities, threshold=0.5):
        # Assuming your labels are in the format: [label1, label2, ...]
        id2label =  self.id2label_parliament

        predicted_labels = [id2label[i] for i, prob in enumerate(probabilities[0]) if prob > threshold]
        return predicted_labels
    
    def decode_predictions_twitter(self, probabilities, threshold=0.5):
        # Assuming your labels are in the format: [label1, label2, ...]
        id2label = self.id2label_twitter

        predicted_labels = [id2label[i] for i, prob in enumerate(probabilities[0]) if prob > threshold]
        return predicted_labels
    