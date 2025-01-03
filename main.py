import os
import sys
import time

import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

repo_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(repo_dir)


def read_config():

    config_file = os.path.join(repo_dir, 'config.yml')

    with open(config_file, 'r') as file:
        config_data = yaml.safe_load(file)

    return config_data


def get_device():
    '''
    Returns the appropriate device (GPU if available, otherwise CPU).
    '''

    if torch.cuda.is_available():
        print("GPU detected. Using GPU for inference.")
        return torch.device('cuda')
    else:
        print("No GPU detected. Using CPU for inference. NOT RECOMMENDED.")
        return torch.device('cpu')


if __name__ == '__main__':

    PROMPT = 'Write a poem about the fall of the Roman Empire.'

    config_data = read_config()
    access_token = config_data['token']
    model_name = config_data['model']

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16)

    cache_dir = os.path.join(repo_dir, 'model')

    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=model_name,
        token=access_token,
        cache_dir=cache_dir)
    
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        token=access_token,
        cache_dir=cache_dir,
        low_cpu_mem_usage=True,
        quantization_config=quantization_config,
        device_map='auto' if device.type=='cuda' else None)
    
    # model.to(device)  # Send the model to the GPU.

    

    inputs = tokenizer(PROMPT, return_tensors='pt')#.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    start_time = time.time()

    outputs = model.generate(**inputs, max_new_tokens=100)
    # with torch.no_grad():
    #     outputs = model.generate(**inputs, max_new_tokens=100)

    elapsed_time = time.time() - start_time
    print(f'Time to generate output: {elapsed_time:.2f} sec')

    outputs_text = tokenizer.decode(outputs[0], skip_special_tokens=True)  # do_sample=False

    # Check model device
    # for param in model.parameters():
    #     print(f"Model is on device: {param.device}")
    #     break  # Only need to check the first parameter

    print(f'Elapsed Time: {elapsed_time:.2f} seconds.')
    print(outputs_text)


