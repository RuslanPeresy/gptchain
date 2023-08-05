import click
from langchain.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

from langchain.llms import LlamaCpp
from langchain import PromptTemplate, LLMChain
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# -------------------
# OpenAI


@click.command()
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


# -------------------
# Llama / other open source models
# https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard


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


if __name__ == '__main__':
    retrieve_openai()
