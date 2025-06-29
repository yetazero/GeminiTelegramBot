import os
import telegram
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold, BlockedPromptException
import re
import asyncio
from PIL import Image
from io import BytesIO
import base64
import logging
import time
import html
from dotenv import load_dotenv
from telegram.constants import ParseMode, ChatAction

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not TELEGRAM_BOT_TOKEN or not GEMINI_API_KEY:
    raise ValueError("TELEGRAM_BOT_TOKEN and GEMINI_API_KEY environment variables must be set. Please check your .env file.")

genai.configure(api_key=GEMINI_API_KEY)

INITIAL_HTML_INSTRUCTION = [
    {
        "role": "user",
        "parts": [{"text": "For our responses, please use HTML markup, but only the following Telegram allowed tags: <b>, <i>, <u>, <s>, <pre>, <code>, <a href=\"URL\">text</a>, <blockquote>. Never use <html>, <body>, <head>, <!DOCTYPE html>, <title> tags."}]
    },
    {
        "role": "model",
        "parts": [{"text": "Understood. I will format responses using only allowed Telegram HTML tags and avoid prohibited tags."}]
    }
]

text_model = genai.GenerativeModel('gemini-1.5-flash')
vision_audio_model = genai.GenerativeModel('gemini-1.5-flash')

safety_settings = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

user_chat_sessions = {}

RATE_LIMIT_SECONDS = 10
RATE_LIMIT_MESSAGES = 1
COOLDOWN_SECONDS = 30

REPEAT_MESSAGE_THRESHOLD = 2
REPEAT_MESSAGE_COOLDOWN = 10

user_activity = {}

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

TELEGRAM_SUPPORTED_HTML_TAGS = {
    'b', 'strong', 'i', 'em', 'u', 'ins', 's', 'strike', 'del',
    'code', 'pre', 'blockquote', 'a'
}

def convert_markdown_to_html(text: str) -> str:
    # Handle code blocks first to prevent inner formatting
    text = re.sub(r'```(.*?)\n(.*?)\n```', r'<pre>\2</pre>', text, flags=re.DOTALL)
    text = re.sub(r'```(.*?)```', r'<pre>\1</pre>', text, flags=re.DOTALL)
    
    # Bold and Italic
    text = re.sub(r'\*\*\*(.*?)\*\*\*', r'<b><i>\1</i></b>', text)
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    
    # Strikethrough
    text = re.sub(r'~~(.*?)~~', r'<s>\1</s>', text)
    
    # Inline code
    text = re.sub(r'`(.*?)`', r'<code>\1</code>', text)
    
    return text

def post_process_gemini_output(raw_gemini_text: str) -> str:
    # Remove <p> and </p> tags as they are not supported by Telegram
    raw_gemini_text = raw_gemini_text.replace('<p>', '').replace('</p>', '')

    parts = []
    last_idx = 0

    tag_pattern = re.compile(r'(<)(\/?)([\w:]+)([^>]*)?>', re.IGNORECASE)

    for match in tag_pattern.finditer(raw_gemini_text):
        tag_start, tag_end = match.span()
        
        if tag_start > last_idx:
            parts.append(html.escape(raw_gemini_text[last_idx:tag_start]))

        full_tag_string = match.group(0)
        is_closing_tag = match.group(2)
        tag_name = match.group(3).lower()
        attributes_string = match.group(4) if match.group(4) else ''

        if tag_name in TELEGRAM_SUPPORTED_HTML_TAGS:
            if tag_name == 'a':
                reconstructed_tag = f"<{is_closing_tag}{tag_name}{attributes_string}>"
            else:
                reconstructed_tag = f"<{is_closing_tag}{tag_name}>"
            parts.append(reconstructed_tag)
        else:
            parts.append(html.escape(full_tag_string))

        last_idx = tag_end
    
    if last_idx < len(raw_gemini_text):
        parts.append(html.escape(raw_gemini_text[last_idx:]))

    return "".join(parts)

async def send_long_message(update, text, parse_mode):
    MAX_MESSAGE_LENGTH = 4000
    
    lines = text.split('\n')
    chunks = []
    current_chunk = ""
    
    for line in lines:
        if len(current_chunk) + len(line) + 1 > MAX_MESSAGE_LENGTH:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = line
        else:
            if current_chunk:
                current_chunk += "\n" + line
            else:
                current_chunk = line
    
    if current_chunk:
        chunks.append(current_chunk)

    for chunk in chunks:
        if chunk.strip():
            await update.message.reply_text(chunk, parse_mode=parse_mode)

async def check_spam(update: Update, message_content: str) -> bool:
    user_id = update.message.from_user.id
    current_time = time.time()

    if user_id not in user_activity:
        user_activity[user_id] = {'timestamps': [], 'last_message': '', 'last_message_count': 0, 'blocked_until': 0}

    activity = user_activity[user_id]

    if activity['blocked_until'] > current_time:
        remaining_time = int(activity['blocked_until'] - current_time)
        await update.message.reply_text(f"Please wait {remaining_time} seconds before sending a new request. This is part of spam protection.")
        logger.info(f"User {user_id} is still blocked for {remaining_time} seconds.")
        return True

    activity['timestamps'] = [t for t in activity['timestamps'] if current_time - t < RATE_LIMIT_SECONDS]
    activity['timestamps'].append(current_time)

    if message_content and message_content.lower() == activity['last_message'].lower() and \
       activity['last_message_count'] >= REPEAT_MESSAGE_THRESHOLD and \
       (current_time - activity['timestamps'][-1] < REPEAT_MESSAGE_COOLDOWN if activity['timestamps'] else False):
        await update.message.reply_text("I have already received this message. Please send something new or wait a bit.")
        logger.info(f"User {user_id} sent identical message repeatedly.")
        return True

    if message_content and message_content.lower() == activity['last_message'].lower():
        activity['last_message_count'] += 1
    else:
        activity['last_message'] = message_content
        activity['last_message_count'] = 1

    if len(activity['timestamps']) > RATE_LIMIT_MESSAGES:
        activity['blocked_until'] = current_time + COOLDOWN_SECONDS
        await update.message.reply_text(f"You are sending too many messages. Please wait {COOLDOWN_SECONDS} seconds.")
        logger.warning(f"User {user_id} hit rate limit, blocked for {COOLDOWN_SECONDS} seconds.")
        return True

    return False

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info(f"User {update.message.from_user.id} started the bot.")
    await update.message.reply_text('Hi! I am a Gemini-powered bot. Send me a message, photo, or voice message, and I will try to respond.')

async def clear_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.message.from_user.id
    if user_id in user_chat_sessions:
        del user_chat_sessions[user_id]
        logger.info(f"Chat history cleared for user {user_id}.")
        await update.message.reply_text("Chat history cleared. We can now start a new conversation.")
    else:
        await update.message.reply_text("Your chat history is already empty.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.message.from_user.id
    user_message = update.message.text
    logger.info(f"User {user_id} sent text message: {user_message}")

    if await check_spam(update, user_message):
        return

    await update.message.chat.send_action(ChatAction.TYPING)

    if user_id not in user_chat_sessions or user_chat_sessions[user_id].model != text_model:
        user_chat_sessions[user_id] = text_model.start_chat(history=INITIAL_HTML_INSTRUCTION)
        logger.info(f"Started new text chat session for user {user_id} with HTML instruction.")

    try:
        response = await user_chat_sessions[user_id].send_message_async(
            user_message,
            safety_settings=safety_settings
        )
        
        html_text = convert_markdown_to_html(response.text)
        final_text_for_telegram = post_process_gemini_output(html_text)
        
        await send_long_message(update, final_text_for_telegram, ParseMode.HTML)
    except BlockedPromptException as e:
        logger.warning(f"Blocked content detected for user {user_id}: {e}")
        await update.message.reply_text("Sorry, your request or response contains content that was blocked due to safety settings.")
    except Exception as e:
        logger.error(f"Error processing text message from {user_id}: {e}", exc_info=True)
        await update.message.reply_text(f"An error occurred while processing your text request: {e}")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.message.from_user.id
    caption_prompt = update.message.caption if update.message.caption else ""
    logger.info(f"User {user_id} sent photo with caption: {caption_prompt}")

    spam_check_content = caption_prompt if caption_prompt else f"photo__{update.message.photo[-1].file_id}"
    if await check_spam(update, spam_check_content):
        return

    await update.message.reply_text("Received, processing...")
    await update.message.chat.send_action(ChatAction.UPLOAD_PHOTO)
    
    file_id = update.message.photo[-1].file_id
    file = await context.bot.get_file(file_id)
    photo_dir = "downloads"
    os.makedirs(photo_dir, exist_ok=True)
    photo_path = os.path.join(photo_dir, f"{file_id}.jpg")

    uploaded_file = None
    
    try:
        await file.download_to_drive(photo_path)
        logger.info(f"Photo downloaded to {photo_path}")

        uploaded_file = genai.upload_file(path=photo_path)
        logger.info(f"Photo uploaded to Gemini: {uploaded_file.name}")

        if user_id not in user_chat_sessions or user_chat_sessions[user_id].model != vision_audio_model:
            user_chat_sessions[user_id] = vision_audio_model.start_chat(history=INITIAL_HTML_INSTRUCTION)
            logger.info(f"Started new vision/audio chat session for user {user_id} with HTML instruction.")
        
        request_content = [uploaded_file, caption_prompt] if caption_prompt else [uploaded_file]
        
        response = await user_chat_sessions[user_id].send_message_async(
            request_content,
            safety_settings=safety_settings
        )
        
        html_text = convert_markdown_to_html(response.text)
        final_text_for_telegram = post_process_gemini_output(html_text)

        await send_long_message(update, final_text_for_telegram, ParseMode.HTML)
    except BlockedPromptException as e:
        logger.warning(f"Blocked content detected for user {user_id}: {e}")
        await update.message.reply_text("Sorry, your request or response contains content that was blocked due to safety settings.")
    except Exception as e:
        logger.error(f"Error processing photo from {user_id}: {e}", exc_info=True)
        await update.message.reply_text(f"An error occurred while processing your photo: {e}")
    finally:
        if os.path.exists(photo_path):
            os.remove(photo_path)
            logger.info(f"Local photo file {photo_path} removed.")
        if uploaded_file and hasattr(uploaded_file, 'name'):
            try:
                await genai.delete_file_async(uploaded_file.name)
                logger.info(f"Gemini file {uploaded_file.name} deleted.")
            except Exception as e:
                logger.error(f"Error deleting Gemini file {uploaded_file.name}: {e}")

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.message.from_user.id
    logger.info(f"User {user_id} sent voice message.")

    spam_check_content = f"voice__{update.message.voice.file_id}"
    if await check_spam(update, spam_check_content):
        return

    await update.message.reply_text("Uploading voice message, please wait...")
    await update.message.chat.send_action(ChatAction.UPLOAD_VOICE) 
    
    file_id = update.message.voice.file_id
    file = await context.bot.get_file(file_id)
    voice_dir = "downloads"
    os.makedirs(voice_dir, exist_ok=True)
    voice_path = os.path.join(voice_dir, f"{file_id}.ogg")

    uploaded_file = None

    try:
        await file.download_to_drive(voice_path)
        logger.info(f"Voice message downloaded to {voice_path}")

        uploaded_file = genai.upload_file(path=voice_path, mime_type="audio/ogg")
        logger.info(f"Voice message uploaded to Gemini: {uploaded_file.name}")
        
        prompt_for_gemini = "Transcribe the following voice message, then respond to its content."
        
        if user_id not in user_chat_sessions or user_chat_sessions[user_id].model != vision_audio_model:
            user_chat_sessions[user_id] = vision_audio_model.start_chat(history=INITIAL_HTML_INSTRUCTION)
            logger.info(f"Started new vision/audio chat session for user {user_id} with HTML instruction.")

        response = await user_chat_sessions[user_id].send_message_async(
            [uploaded_file, prompt_for_gemini],
            safety_settings=safety_settings
        )
        
        html_text = convert_markdown_to_html(response.text)
        final_text_for_telegram = post_process_gemini_output(html_text)

        await send_long_message(update, final_text_for_telegram, ParseMode.HTML)
    except BlockedPromptException as e:
        logger.warning(f"Blocked content detected for user {user_id}: {e}")
        await update.message.reply_text("Sorry, your request or response contains content that was blocked due to safety settings.")
    except Exception as e:
        logger.error(f"Error processing voice message from {user_id}: {e}", exc_info=True)
        await update.message.reply_text(f"An error occurred while processing your voice message: {e}")
    finally:
        if os.path.exists(voice_path):
            os.remove(voice_path)
            logger.info(f"Local voice file {voice_path} removed.")
        if uploaded_file and hasattr(uploaded_file, 'name'):
            try:
                await genai.delete_file_async(uploaded_file.name)
                logger.info(f"Gemini file {uploaded_file.name} deleted.")
            except Exception as e:
                logger.error(f"Error deleting Gemini file {uploaded_file.name}: {e}")

async def unhandled_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.message.from_user.id
    content_type = update.message.effective_attachment.content_type if update.message.effective_attachment else "unknown"
    logger.info(f"User {user_id} sent unhandled message type: {content_type}")

    spam_check_content = f"unhandled__{content_type}"
    if await check_spam(update, spam_check_content):
        return

    reply_text = (
        "Sorry, I cannot currently process messages of type "
        f"'{content_type}'.\n\n"
        "Please send me a text message, photo, or voice message."
    )
    await update.message.reply_text(reply_text)

def main() -> None:
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("clear", clear_history))
    
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice))
    application.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND, unhandled_message))

    logger.info("Bot started. Send /start, /clear, any message, photo or voice message in Telegram.")
    application.run_polling(allowed_updates=Update.ALL_TYPES)
    logger.info("Bot stopped.")

if __name__ == '__main__':
    main()
