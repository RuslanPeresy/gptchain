import os
from telegram.ext import Application, MessageHandler, filters

from gptchain import get_rag_chain, get_llm_chain_local

TOKEN = os.getenv('BOT_TOKEN')
INFERENCE_URL = ''


async def handle_message(update, context):
    text = update.message.text
    print(f'User: {text}')

    llm_chain = get_llm_chain_local()
    response = llm_chain.run(text)
    print(f'Bot: {response}')

    await update.message.reply_text(response)


async def handle_document(update, context):
    file = await context.bot.get_file(update.message.document)
    caption = update.message.caption
    print(f'User sent file - {file}')
    print(f'Caption: {caption}')
    data_path = await file.download_to_drive()

    chain = get_rag_chain(INFERENCE_URL, str(data_path))
    response = chain(caption)
    response = response['answer'].strip()
    print(f'Bot: {response}')

    await update.message.reply_text(response)


async def error(update, context):
    print(context.error)


if __name__ == '__main__':
    print('Set inference endpoint:')
    INFERENCE_URL = input()
    app = Application.builder().token(TOKEN).build()

    app.add_handler(MessageHandler(filters.TEXT, handle_message))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    app.add_error_handler(error)

    app.run_polling(poll_interval=1)
