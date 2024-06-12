from datasets import load_dataset
from unsloth.chat_templates import get_chat_template

from .prompts import system_prompts, alpaca_prompt, vicuna_prompt


class Dataset:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._samantha_data = None
        self._tagengo_gpt4 = None
        self._tagengo_subset_gpt4o = None

    @property
    def samantha_data(self):
        # Samantha dataset https://huggingface.co/datasets/cognitivecomputations/samantha-data
        # in Alpaca format
        if self._samantha_data:
            return self._samantha_data

        EOS_TOKEN = self.tokenizer.eos_token  # Must add EOS_TOKEN

        def formatting_prompts_func(examples):
            conversations = examples['conversations']
            system = system_prompts['samantha']
            texts = []
            for conv in conversations:
                # Must add EOS_TOKEN, otherwise your generation will go on forever!
                text = alpaca_prompt.format(system, conv['human'], conv['gpt']) + EOS_TOKEN
                texts.append(text)
            return {"text": texts, }

        dataset = load_dataset('wangqi777/samantha-data', 'en', split='train')
        self._samantha_data = dataset.map(formatting_prompts_func, batched=True)
        return self._samantha_data

    @property
    def tagengo_gpt4(self):
        # Tagengo - the world's largest high quality multilingual chat dataset
        # https://huggingface.co/datasets/lightblue/tagengo-gpt4
        if self._tagengo_gpt4:
            return self._tagengo_gpt4

        dataset = load_dataset("lightblue/tagengo-gpt4", split="train")
        dataset = dataset.filter(lambda x: x['conversations'][1]["value"])
        self._tagengo_gpt4 = self._sharegpt_to_chatml(dataset)
        return self._tagengo_gpt4

    @property
    def tagengo_subset_gpt4o(self):
        if self._tagengo_subset_gpt4o:
            return self._tagengo_subset_gpt4o

        dataset = load_dataset("ruslandev/tagengo-subset-gpt-4o", split="train")
        dataset = dataset.filter(lambda x: x['conversations'][1]["value"])
        self._tagengo_subset_gpt4o = self._sharegpt_to_chatml(dataset)
        return self._tagengo_subset_gpt4o

    def _sharegpt_to_chatml(self, dataset):
        tokenizer = get_chat_template(
            self.tokenizer,
            chat_template="chatml",  # Supports zephyr, chatml, mistral, llama, alpaca, vicuna, vicuna_old, unsloth
            mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},  # ShareGPT style
            map_eos_token=True,  # Maps <|im_end|> to </s> instead
        )

        def formatting_prompts_func(examples):
            convos = examples["conversations"]
            texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in
                     convos]
            return {"text": texts, }

        return dataset.map(formatting_prompts_func, batched=True)

    def __getitem__(self, dataset_id):
        return getattr(self, dataset_id)
