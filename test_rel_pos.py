from transformers import AutoTokenizer, AutoModelForMaskedLM
from transformers import XLMRobertaConfig, XLMRobertaModel

if __name__=="__main__":
    # get xlmroberta model

    # Initializing a XLM-RoBERTa xlm-roberta-base style configuration
    #configuration = XLMRobertaConfig(position_embedding_type='relative_key')

    # Initializing a model (with random weights) from the xlm-roberta-base style configuration
    model = XLMRobertaModel.from_pretrained('xlm-roberta-base', position_embedding_type='relative_key')


    tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')

    # prepare input
    text = "I am an input"
    encoded_input = tokenizer(text, return_tensors='pt')

    # forward pass
    output = model(**encoded_input)
