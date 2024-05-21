# gptchain
[![slack badge](https://img.shields.io/badge/Discord-join-blueviolet?logo=discord&amp)](https://discord.gg/HXs2tm9)

This project has evolved:

...from a command line application to run Large Language Models (such as OpenAI and Llama) on custom data,
built for the educational YouTube [video](https://youtu.be/tOHdSMELLAQ)

...to a framework with LLM finetuning and deployment capabilities. It supports a few datasets for finetuning out of the box, others can be added easily.

The framework utilises LangChain, Unsloth and TRL.  

# How to use
Clone this repo, then

```
cd gptchain
pip install -r requirements-train.txt
```

## LLM inference

Using OpenAI-like JSON string with automatic ChatML conversion:

```
python gptchain.py chat -m ruslandev/llama-3-70b-tagengo \
	--chatml true \
	-q '[{"from": "human", "value": "Из чего состоит нейронная сеть?"}]'
```

## LLM fine-tune

If you want to upload your model weights to Huggingface after training, create .env file and define HF_TOKEN variable in it with your Huggungface token.

You can also put WANDB_API_KEY there to track training metrics in wandb.

Run training session:

```
python gptchain.py train -m unsloth/llama-3-70b-bnb-4bit \
	--dataset-name tagengo_gpt4 \
	--save-path checkpoints/llama-3-70b-tagengo \
	--huggingface-repo llama-3-70b-tagengo \
	--max-steps 2400
```
Here, the base model is `unsloth/llama-3-70b-bnb-4bit`, dataset is `tagengo_gpt4`, final checkpoint will be stored in `checkpoints/llama-3-70b-tagengo`, weights will be uploaded in the Huggingface repo `llama-3-70b-tagengo` under your namespace. Maximum training steps is 2400, you can pass `--num-epochs` argument instead to set the number of training epochs.

## LLM quantization

You can quantize your model and store it in `gguf` format.

```
python gptchain.py quant -m checkpoints/llama-3-70b-tagengo \
	--method q4_k_m \
	--save-path quants/llama-3-70b-tagengo \
	--huggingface-repo llama-3-70b-tagengo-GGUF
```

Quantization method used here is `q4_k_m`. [All available options you can see here](https://github.com/unslothai/unsloth/wiki#3-gguf-conversion)

## LLM deployment

You can deploy your model to runpod.io with Text Generaion Inference.
Change this variable in deploy/config.py:

```
POD_CONF = {
    "name": "gptchain-0.0.2",
    "image_name": "ghcr.io/huggingface/text-generation-inference:latest",
    "gpu_type_id": "NVIDIA RTX A6000",
    "cloud_type": "SECURE",
    "docker_args": f"--model-id l3utterfly/llama2-7b-layla --num-shard {GPU_COUNT}",
    "gpu_count": GPU_COUNT,
    "volume_in_gb": 50,
    "container_disk_in_gb": 20,
    "ports": "80/http,29500/http",
    "volume_mount_path": "/data",
}
```

Here you can set your pod `name`, GPU type - `gpu_type_id`, and other parameters.
To deploy your model with the defined configuration, use this command:

```
python gptchain.py deploy-model -m ruslandev/llama-3-70b-tagengo
```

## Retrieval Augmented Generation

To run RAG pipeline on custom data, you need to deploy an LLM with TGI server first (see the section above).

Prepare your data - let's say a file `mydata.txt`

Then you can use this command:

```
python gptchain.py rag --inference-url 'https://${POD_ID}-80.proxy.runpod.net' \
	--data-path mydata.txt \
	-q 'What is this text about?'
```
