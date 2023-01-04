import os, sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(PROJECT_ROOT)

import torch

def sentiment_analyzer(sentence, model, tokenizer):
    '''
        Returns list of sentiment review as number of stars from 1 - 5.
        1 - Star: Very Negative
        2 - Star: Negative
        3 - Star: Neutral
        4 - Star: Positive
        5 - Star: Very Positive 

        Input:
            sentence: str
            model: Pretrained model can be original transformer or in onnx form
            tokenizer: Pretrained tokenizer
        
        Output:
            results: list of dictionaries. Each dictionary consist of star and score key value pair. 
    '''
    batch = tokenizer(sentence, padding=True, truncation=True, max_length=512, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**batch)
        predictions = torch.softmax(outputs.logits, dim=1)[0].tolist()
        results =  [{f'{label+1} star': round(float(result), 4)} for label, result in enumerate(predictions)]
        
    return results
