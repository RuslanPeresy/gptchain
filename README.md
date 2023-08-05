# gptchain

An command line application to transfer any custom data to a language model (ChatGPT) and get response based on that data. The application utilises LangChain library.  
It is made for my YouTube [video](https://youtu.be/tOHdSMELLAQ)

# Setup
```pip install -r requirements.txt```

if you want llama-cpp support:

for CPU:

```pip install llama-cpp-python```

for GPU, install with one if the BLAS backends support, eg. cuBLAS:

```CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python```

for Apple Silicon:

```CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip install llama-cpp-python```

[More info](https://python.langchain.com/docs/integrations/llms/llamacpp#installation)

# Usage
1. Replace data.txt with your data.
2. ```python gptchain.py -q "Insert your query here"```
