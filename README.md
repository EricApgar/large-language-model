# large-language-model
Interact with an LLM in an easy to user interface that gives you control over the most needed aspects. This library is designed to support multiple LLMs from [Hugging Face](https://huggingface.co/models?pipeline_tag=text-generation&sort=trending) with a standard interface.

# Setup
See [Releases](https://github.com/EricApgar/large-language-model/releases) to install from wheel file.  

See ```pyproject.toml``` for required Python version and dependencies.

## Optional Dependencies
This library uses optional dependencies so you aren't forced to download a lot of supporting libraries that are only needed for a single model if you know you only need to use one specific LLM. See the ```pyproject.toml``` for the list of optional library tags based on the model type. To just install all options, use the ```all``` tag.

## Library
Install this repo as a library into an another project.

### ...with uv:
```
uv add "llm[all] @ git+https://github.com/EricApgar/large-language-model"
```

...with optional libraries:
```
uv add "llm[<tag-1>,<tag-2>] @ git+https://github.com/EricApgar/large-language-model"
```
Example:  
```uv add "llm[microsoft] @ git+https://github.com/EricApgar/large-language-model"```  

### ...with pip.
```
pip install "llm[all] @ git+https://github.com/EricApgar/large-language-model"
```

...with optional libraries:
```
pip install "llm[<tag-1>,<tag-2>] @ git+https://github.com/EricApgar/large-language-model"
```
Example:  
```pip install "llm[microsoft] @ git+https://github.com/EricApgar/large-language-model"```  

## Repo
Run locally for development of this repo.  
Create a virtual environment and then install the dependencies into the environment.

### ...with uv:
```
uv sync
```

...with optional libraries:
```
uv sync --extra <tag-1> <tag-2>
```

### ...with pip:
```
pip install -r requirements.txt
```

...with optional libraries:
```
pip install -e ".[<tag-1>]"
```

## Hardware
Running an LLM requires an NVIDIA GPU with ideally a large number of TOPS. While it technically runs on the CPU, inference times are unusably long.

This repo has been tested on an **NVIDIA RTX 5070 Ti** with Linux Mint. Other GPUs may see different run times. Windows capabilities not guranteed.

### Update GPU Drivers
For help updating the GPU drivers, see the [wiki](https://github.com/EricApgar/large-language-model/wiki/Update-Nvidia-Drivers)

# Usage
See the [wiki](https://github.com/EricApgar/large-language-model/wiki/Usage) for full details.

As a library...

```
import llm


model = llm.model(name='openai/gpt-oss-20b')
model.load(location=<path to model cache dir>)

response = model.ask(prompt='Tell me a joke.')

print(response)
```

# Supported Models
| Model | Supported Inputs |
|-|-|
| [openai/gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b) | text |
| [microsoft/Phi-4-multimodal-instruct](https://huggingface.co/microsoft/Phi-4-multimodal-instruct) | text, images |