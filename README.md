# large-language-model
Interact with an LLM.

# Python
Python 3.12.7

# Update GPU Drivers
For help updating the GPU drivers, see the [wiki](https://github.com/EricApgar/large-language-model/wiki/Update-Nvidia-Drivers)

# Installing the right version of Cuda:
To make use of your GPU you need to install the proper version of PyTorch and Cuda support.

You can double check that the version of PyTorch you have installed is the latest (or a previous version) by running:
```
python -c "import torch; print(torch.__version__)"
```

Check the [PyTorch](https://pytorch.org/) website and use their version selector to get the download command.

| Field | Value |
|-|-|
| PyTorch Build | Stable |
| OS | your OS |
| Package | Pip |
| Compute Platform | CUDA 12.4 |

# CPU vs GPU
This will run on a CPU, but it runs so slowly that it's not a practical solution. So, while useful to know that it will work on the without one, it's not recommended that you run this unless you have a GPU available to use.

## Time Comparison 

| Device | Time |
|-|-|
| GPU Nvidia 3060 Ti | 52 sec |
| CPU AMD Ryzen 5 3600 | 1 hr, 15 min |