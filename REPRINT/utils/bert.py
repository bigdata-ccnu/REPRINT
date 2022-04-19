import torch
import numpy as np
from transformers import BertModel, BertTokenizer
from transformers import AlbertModel, AlbertTokenizer
from tqdm import tqdm

# model_class = BertModel
# tokenizer_class = BertTokenizer
# pretrained_weights = 'bert-base-uncased'
# tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
# model = model_class.from_pretrained(pretrained_weights).to("cuda:0" if torch.cuda.is_available() else "cpu")

tokenizer_class = AlbertTokenizer.from_pretrained('albert-base-v2')
model_class = AlbertModel.from_pretrained("albert-base-v2")
pretrained_weights = 'albert-base-v2'
tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
model = model_class.from_pretrained(pretrained_weights).to("cuda:0" if torch.cuda.is_available() else "cpu")

# Encode text
def get_embedding(input_string):
    input_ids = tokenizer.encode(input_string, add_special_tokens = True)
    if len(input_ids) > 512:
        input_ids = input_ids[:512]
    input_ids = torch.tensor([input_ids]).to("cuda:0" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0].cpu().numpy()
        last_hidden_states = np.mean(last_hidden_states, axis = 1)
        sentence_embedding = last_hidden_states.flatten()
        return sentence_embedding
