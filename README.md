# large-language-model
Interact with an LLM in an easy to user interface that gives you control over the most needed aspects. This library is designed to support multiple LLMs from [Hugging Face](https://huggingface.co/models?pipeline_tag=text-generation&sort=trending) with a standard interface.

# Setup
See [Releases](https://github.com/EricApgar/large-language-model/releases) to install from wheel file.  
See ```pyproject.toml``` for required Python version and dependencies.

## Hardware
Running an LLM requires an NVIDIA GPU with ideally a large number of TOPS. While it technically runs on the CPU, inference times are unusably long.

This repo has been tested on an **NVIDIA RTX 5070 Ti** with Linux Mint. Other GPUs may see different run times. Windows capabilities not guranteed.

### Update GPU Drivers
For help updating the GPU drivers, see the [wiki](https://github.com/EricApgar/large-language-model/wiki/Update-Nvidia-Drivers)

# Usage
As a library...

```
import llm


model = llm.model(name='openai/gpt-oss-20b', hf_token=<Hugging Face token>)
model.load(
    location=<path to model cache dir>, 
    remote=True,
    commit=<specific git commit>)
response = model.ask(prompt='Tell me a joke.', max_token=256)

print(response)
```

# Supported Models
| Model |
|-|
| [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) |