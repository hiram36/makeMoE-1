
import torch

from modelscope import snapshot_download, Model
from modelscope.models.nlp.llama2 import Llama2Tokenizer


model_dir = snapshot_download("AI-ModelScope/Mistral-7B-Instruct-v0.2", revision = "master")
'''
model_dir = snapshot_download("modelscope/Llama-2-7b-chat-ms", revision='v1.0.2', 
                              ignore_file_pattern=[r'.+\.bin$'])
'''
tokenizer = Llama2Tokenizer.from_pretrained(model_dir)
model = Model.from_pretrained(
    model_dir,
    torch_dtype=torch.float16,
    device_map='auto'
)

system = 'you are a helpful assistant!'
inputs = {
    'text': 'Where is the capital of Zhejiang?', 
    'system': system, 
    'max_length': 512
}
output = model.chat(inputs, tokenizer)
print(output['response'])

inputs = {
    'text': 'What are the interesting places there?', 
    'system': system, 
    'history': output['history'],
    'max_length': 512
}
output = model.chat(inputs, tokenizer)
print(output['response'])
