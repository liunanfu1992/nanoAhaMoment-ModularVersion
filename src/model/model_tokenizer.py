from typing import Tuple
from transformers import AutoTokenizer
from src.config.configs import MODEL_CONFIG

class ModelTokenizer:
    def __init__(self):
        self.ModelTokenizer=AutoTokenizer.from_pretrained(MODEL_CONFIG["model_name"])
        self.ModelChatTokenizer=AutoTokenizer.from_pretrained(MODEL_CONFIG["model_chat_name"])
    
    def getModelTokenizer(self):
        return self.ModelTokenizer
    
    def getModelChatTokenizer(self):
        return self.ModelChatTokenizer

    def get_eos_token_and_id(self)->Tuple[int,str]:
        tokenizer=self.getModelChatTokenizer()
        EOS_TOKEN_ID = self.getModelTokenizer().eos_token_id
        EOS_TOKEN = tokenizer.convert_ids_to_tokens(EOS_TOKEN_ID)
        return EOS_TOKEN_ID,EOS_TOKEN