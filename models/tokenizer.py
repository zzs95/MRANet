from transformers import GPT2Tokenizer
def get_gpt_tokenizer():
    checkpoint = "healx/gpt-2-pubmed-medium"
    tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

from transformers import AutoTokenizer
def get_bert_tokenizer():
    checkpoint = "emilyalsentzer/Bio_ClinicalBERT"
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    return tokenizer
