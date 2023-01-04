import os, sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(PROJECT_ROOT)

from sentence_transformers import util
import torch

def sentence_embeddings(sentences, model, tokenizer):
    """
    Returns vector embeddings of one or multiple sentences.
    
    Input:
      sentence(s): str
      model: Pretrained model can be original transformer or in onnx form
      tokenizer: Pretrained tokenizer
    
    Output:
      embedding(s): torch.float32
    """
    # Mean Pooling - Take attention mask into account for correct averaging
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[
            0
        ]  # First element of model_output contains all token embeddings
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        )
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
            input_mask_expanded.sum(1), min=1e-9
        )

    # Tokenize sentences
    encoded_input = tokenizer(
        sentences, padding=True, truncation=True, return_tensors="pt"
    )

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling. In this case, max pooling.
    embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
    return embeddings


def sentences_similarity(sentences, model, tokenizer):
    """
    Returns cosine similarities between source sentence which is the first
    sentences and one or multiple sentences to be compared which are passed
    as second argument.
    
    Input:
      sentences: str
      model: Pretrained model can be original transformer or in onnx form
      tokenizer: Pretrained tokenizer
    
    Output:
      cosine_similarities: list of dictionaries consisting of pairs of
      sentences and cosine similarities against first sentence argument
    """
    if len(sentences) >= 2:
        source_sentence = sentences[0]
        sentences_to_be_compared = list(sentences[1:])
        emb1 = sentence_embeddings(source_sentence, model, tokenizer)
        emb2 = sentence_embeddings(sentences_to_be_compared, model, tokenizer)

        # Computes cosine similarities between source sentence and sentence(s) to be compared
        cosine_similarities = util.cos_sim(emb1, emb2)[0].tolist()
        list_cosine_similarities = []
        for sentence_to_be_compared, cosine_similarity in zip(
            sentences_to_be_compared, cosine_similarities
        ):
            list_cosine_similarities.append(
                {sentence_to_be_compared: round(float(cosine_similarity), 3)}
            )
        return list_cosine_similarities
    else:
        return "Error. User must enter more than one sentences."


