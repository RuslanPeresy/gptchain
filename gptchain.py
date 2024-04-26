import click
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, RetrievalQAWithSourcesChain, LLMChain

from langchain.llms import LlamaCpp, HuggingFaceTextGenInference
from langchain.prompts import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from unsloth import FastLanguageModel

from deploy.runpod import deploy_llm
from train import train_model
from utils.data import Dataset
from utils.weights import load_model_4bit, apply_lora, max_seq_length
from utils.prompts import system_prompts, alpaca_prompt, vicuna_prompt


@click.group()
def cli():
    pass

# -------------------
# OpenAI


@cli.command('retrieve-openai')
@click.option('--query', '-q', required=True)
def retrieve_openai(query):
    loader = TextLoader('data.txt')
    index = VectorstoreIndexCreator().from_loaders([loader])
    chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model='gpt-3.5-turbo'),
        retriever=index.vectorstore.as_retriever(
            search_kwargs={'k': 1}
        )
    )
    chat_log = []
    result = chain({'question': query, 'chat_history': chat_log})
    click.echo(result['answer'])

    chat_log.append((query, result['answer']))


@cli.command('deploy-model')
@click.option('--model', '-m')
def deploy_model(model):
    endpoint = deploy_llm(model_id=model)
    click.echo(f'Use this endpoint: {endpoint}')


# -------------------
# Llama / other open source models
# https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard
def get_llm_chain_local():
    n_gpu_layers = 1  # Metal set to 1 is enough.
    # Should be between 1 and n_ctx, consider the amount of RAM of your Apple Silicon Chip.
    n_batch = 512

    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    # Make sure the model path is correct for your system!
    llm = LlamaCpp(
        model_path="./models/llama-2-7b-chat.ggmlv3.q4_0.bin",
        n_gpu_layers=n_gpu_layers,
        n_batch=n_batch,
        f16_kv=True,  # MUST set to True, otherwise you will run into problem after a couple of calls
        # callback_manager=callback_manager,
        # verbose=True,
    )

    template = """Question: {question}

    Answer: Let's work this out in a step by step way to be sure we have the right answer."""

    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=llm)
    return llm_chain


def get_rag_chain(inference_server_url, data_path):
    llm = HuggingFaceTextGenInference(
        inference_server_url=inference_server_url,
        stop_sequences=['User:'],
        max_new_tokens=500,
        top_k=10,
        top_p=0.95,
        typical_p=0.95,
        temperature=0.1,
        repetition_penalty=1.03
    )

    prompt_template = PromptTemplate(
        input_variables=['summaries', 'question'],
        template='''{summaries}
User: {question}
Assistant:'''
    )
    loader = TextLoader(data_path)
    index = VectorstoreIndexCreator().from_loaders([loader])
    retriever = index.vectorstore.as_retriever(
        search_kwargs={'k': 1}
    )

    return RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=retriever,
        chain_type_kwargs={
            'prompt': prompt_template
        }
    )


@cli.command('rag')
@click.option('--inference-url', '-iu', required=True)
@click.option('--data-path', '-dp', required=True)
@click.option('--question', '-q', required=True)
def rag(inference_url, data_path, question):
    chain = get_rag_chain(inference_url, data_path)
    response = chain(question)
    click.echo(response['answer'].strip())


@cli.command('train')
@click.option('--model_id', '-m', default="unsloth/llama-3-8b-bnb-4bit", help='HF id or path to checkpoint')
@click.option('--dataset-name', '-dn', default="samantha_data")
@click.option('--save-path', '-sp', required=True)
@click.option('--huggingface-repo', '-hf')
@click.option('--max-steps', '-ms', default=60)
@click.option('--num-epochs', '-ne', type=int, help='Number of training epoches, max-steps will be ignored')
@click.option('--no-lora', '-nl', type=bool, help='Apply/do not apply LoRA')
def train(model_id, dataset_name, save_path, huggingface_repo, max_steps, num_epochs, no_lora):
    model, tokenizer = load_model_4bit(model_id)
    if not no_lora:
        model = apply_lora(model)
    data = Dataset(tokenizer)
    train_args = {}
    if num_epochs:
        train_args['num_train_epochs'] = num_epochs
    else:
        train_args['max_steps'] = max_steps
    train_model(model, tokenizer, data[dataset_name], max_seq_length, train_args)
    model.save_pretrained(save_path)
    click.echo(f'Model saved to {save_path}')
    if huggingface_repo:
        click.echo(f'Pushing model to HuggingFace Hub...')
        model.push_to_hub(huggingface_repo)


@cli.command('chat')
@click.option('model_id', '-m', required=True, help='HF id or path to checkpoint')
@click.option('--question', '-q', required=True)
def chat(model_id, question):
    model, tokenizer = load_model_4bit(model_id)
    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
    inputs = tokenizer(
        [
            vicuna_prompt.format(
                system_prompts['samantha'],  # system
                question,  # input
                "",  # output - leave this blank for generation!
            )
        ], return_tensors="pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=64, use_cache=True)
    results = tokenizer.batch_decode(outputs)
    click.echo(results[0].split('### Assistant:')[-1].strip())


if __name__ == '__main__':
    cli()
