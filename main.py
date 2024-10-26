import os
import sys

import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM


repo_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(repo_dir)


def read_config():

    config_file = os.path.join(repo_dir, 'config.yml')

    with open(config_file, 'r') as file:
        config_data = yaml.safe_load(file)

    return config_data


if __name__ == '__main__':
    
    PROMPT = 'Write a poem about the fall of the Roman Empire.'

    config_data = read_config()
    access_token = config_data['access_token']
    model_name = config_data['model']

    cache_dir = os.path.join(repo_dir, 'model')

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_name,
        token=access_token,
        cache_dir=cache_dir)
    
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        token=access_token,
        cache_dir=cache_dir)
    
    inputs = tokenizer(PROMPT, return_tensors='pt')

    outputs = model.generate(**inputs, max_new_tokens=100)

    outputs_text = tokenizer.decode(outputs[0])

    print(outputs_text)
