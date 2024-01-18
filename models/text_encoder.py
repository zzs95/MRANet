import os
import numpy as np
import torch
os.environ["HF_ENDPOINT"] = "https://huggingface.co"

from transformers import AutoModel, AutoTokenizer, AutoConfig

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_model():
    MODEL_NAME = 'UFNLP/gatortron-medium' # 2560
    # MODEL_NAME = 'UFNLP/gatortron-base' # 1024
    model = AutoModel.from_pretrained(MODEL_NAME).cuda()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    return model, tokenizer

class text_feat_encoder:
    def __init__(self,):
        super().__init__()
        self.model, self.tokenizer = get_model()
    def forward(self, sentences_list):
        encoded_input = self.tokenizer(sentences_list, padding=True, return_tensors='pt', truncation=True, max_length=256 )
        for k in encoded_input.keys():
            encoded_input[k] = encoded_input[k].cuda()
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            report_feat = sentence_embeddings.cpu().detach().numpy().squeeze()
        return report_feat