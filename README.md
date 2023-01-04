# Conversion of Transformer Models to ONNX (Open Neural Network eXchange) with 🤗 Optimum 

## Brief Introduction

Inferencing using the models in various frameworks such Tensorflow, PyTorch and Hugging Face can be slow and inefficient. Therefore, these models needs to be first converted into a serialized format that can be loaded, optimized and executed on specialized runtime environments before deploying them. The model in this serialized format is known as ONNX. 

ONNX is general standard for representing Machine/Deep Learning models that can be used for variety of frameworks such as PyTorch, Tensorflow and Hugging Face. We will be using open source library, 🤗 Optimum, for achieving maximum efficiency to train and run models. 

In this tutorial, we will not further deep dive into technical aspects of ONNX. Main aim of this project is to get started with conversion of transformer models to ONNX models. We will see how 🤗 Transformer will be converted into ONNX model and then we will see how we will do inferencing. Note that inferencing is done on CPU as PyTorch GPU is not included in `requirements.txt` file.

## Setting up Environment

1. Clone the entire repository. 
```bash
git clone https://github.com/di37/convert-transformers-to-onnx.git
```

2. Create conda environment. 
```bash
conda create -n onnx_runtime_environment python=3.10
```

3. Activate the environment.
```bash
conda activate onnx_runtime_environment
```

4. Install the dependencies.
```bash
pip install -r requirements.txt --no-cache-dir
```

5. For fair comparison between the speed of original transformer model and after being converted into ONNX format, we will be downloading 
the transformer models in our local drive. The below command will allow to download these models in `saved_models` folder.
```bash
python download_models.py
```
Ensure that the models are stored in following file structure.

```bash
saved_models
├── all-MiniLM-L6-v2
│   ├── 1_Pooling
│   │   └── config.json
│   ├── 2_Normalize
│   ├── config.json
│   ├── config_sentence_transformers.json
│   ├── modules.json
│   ├── pytorch_model.bin
│   ├── README.md
│   ├── sentence_bert_config.json
│   ├── special_tokens_map.json
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   └── vocab.txt
└──bert-base-multilingual-uncased-sentiment
    ├── config.json
    ├── pytorch_model.bin
    ├── special_tokens_map.json
    ├── tokenizer_config.json
    ├── tokenizer.json
    └── vocab.txt
```

6. All of the details for conversion of onnx models are included in `convert_to_onnx.ipynb`.

## References
1. Convert Transformers to ONNX with Hugging Face Optimum - https://huggingface.co/blog/convert-transformers-to-onnx
2. Accelerated Inference with Optimum and Transformers Pipelines - https://huggingface.co/blog/optimum-inference
3. Accelerate Transformer inference on CPU with Optimum and ONNX - https://www.youtube.com/watch?v=_AKFDOnrZz8