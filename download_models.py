import os

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_path = 'saved_models'    

# Downloading DistilBert Model
model_path_ds = os.path.join(model_path, "bert-base-multilingual-uncased-sentiment")
model_id = 'nlptown/bert-base-multilingual-uncased-sentiment'
# Loading tokenizer from hugging face hub
tokenizer = AutoTokenizer.from_pretrained(model_id)
# Loading model from hub. Note that as per the different tasks, the model attribute will be changed.
# Therefore, in general -> AutoModelForXXX.
model = AutoModelForSequenceClassification.from_pretrained(model_id) 
tokenizer.save_pretrained(model_path_ds)
model.save_pretrained(model_path_ds)


# Downloading DistilBert Model
model_path_ss = os.path.join(model_path, "all-MiniLM-L6-v2")
model_id = 'sentence-transformers/all-MiniLM-L6-v2'
# Loading tokenizer and model in one go for sentence transformer models
model = SentenceTransformer(model_id)
model.save(model_path_ss)

