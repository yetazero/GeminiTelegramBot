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
import httpx

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not TELEGRAM_BOT_TOKEN or not GEMINI_API_KEY:
    raise ValueError("TELEGRAM_BOT_TOKEN and GEMINI_API_KEY environment variables must be set. Please check your .env file.")

genai.configure(api_key=GEMINI_API_KEY)

INITIAL_HTML_INSTRUCTION = []

text_model = genai.GenerativeModel('gemini-2.5-flash')
vision_audio_model = genai.GenerativeModel('gemini-2.5-flash')

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

def clean_text_formatting(text: str) -> str:
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    return text

def sanitize_html_tags(text: str) -> str:
    return html.escape(text)

async def send_long_message(update, text, parse_mode=None):
    MAX_MESSAGE_LENGTH = 4000
    
    if len(text) <= MAX_MESSAGE_LENGTH:
        try:
            await update.message.reply_text(text, parse_mode=parse_mode)
            return
        except Exception as e:
            logger.warning(f"Failed to send message: {e}")
            await update.message.reply_text(text)
            return
    
    lines = text.split('\n')
    chunks = []
    current_chunk = ""
    
    for line in lines:
        if len(current_chunk) + len(line) + 1 > MAX_MESSAGE_LENGTH:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = line
        else:
            if current_chunk:
                current_chunk += "\n" + line
            else:
                current_chunk = line
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    for chunk in chunks:
        if chunk:
            try:
                await update.message.reply_text(chunk, parse_mode=parse_mode)
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.warning(f"Failed to send chunk: {e}")
                await update.message.reply_text(chunk)
                await asyncio.sleep(0.5)

async def check_spam(update: Update, message_content: str) -> bool:
    user_id = update.message.from_user.id
    current_time = time.time()
    
    if user_id not in user_activity:
        user_activity[user_id] = {
            'timestamps': [],
            'last_message': '',
            'last_message_count': 0,
            'blocked_until': 0
        }
    
    activity = user_activity[user_id]
    
    if activity['blocked_until'] > current_time:
        remaining_time = int(activity['blocked_until'] - current_time)
        await update.message.reply_text(
            f"Please wait {remaining_time} seconds before sending a new request. "
            "This is part of spam protection."
        )
        logger.info(f"User {user_id} is still blocked for {remaining_time} seconds.")
        return True
    
    activity['timestamps'] = [t for t in activity['timestamps']
                              if current_time - t < RATE_LIMIT_SECONDS]
    activity['timestamps'].append(current_time)
    
    if (message_content and
        message_content.lower() == activity['last_message'].lower() and
        activity['last_message_count'] >= REPEAT_MESSAGE_THRESHOLD):
        await update.message.reply_text(
            "I have already received this message. Please send something new or wait a bit."
        )
        logger.info(f"User {user_id} sent identical message repeatedly.")
        return True
    
    if message_content and message_content.lower() == activity['last_message'].lower():
        activity['last_message_count'] += 1
    else:
        activity['last_message'] = message_content
        activity['last_message_count'] = 1
    
    if len(activity['timestamps']) > RATE_LIMIT_MESSAGES:
        activity['blocked_until'] = current_time + COOLDOWN_SECONDS
        await update.message.reply_text(
            f"You are sending too many messages. "
            f"Please wait {COOLDOWN_SECONDS} seconds."
        )
        logger.warning(f"User {user_id} hit rate limit, blocked for {COOLDOWN_SECONDS} seconds.")
        return True
    
    return False

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info(f"User {update.message.from_user.id} started the bot.")
    await update.message.reply_text(
        'Hi! I am a Gemini-based bot. Send me a message, photo, or voice message, and I will try to respond.'
    )

async def clear_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.message.from_user.id
    if user_id in user_chat_sessions:
        del user_chat_sessions[user_id]
        logger.info(f"Chat history cleared for user {user_id}.")
        await update.message.reply_text("Chat history cleared. We can start a new conversation now.")
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
        logger.info(f"Started new text chat session for user {user_id}.")
    
    try:
        response = await user_chat_sessions[user_id].send_message_async(
            user_message,
            safety_settings=safety_settings
        )
        
        final_text = clean_text_formatting(response.text)
        
        await send_long_message(update, final_text)
        
    except BlockedPromptException as e:
        logger.warning(f"Blocked content detected for user {user_id}: {e}")
        await update.message.reply_text(
            "Sorry, your request or response contains content that was blocked due to safety settings."
        )
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
            logger.info(f"Started new vision/audio chat session for user {user_id}.")
        
        request_content = [uploaded_file, caption_prompt] if caption_prompt else [uploaded_file]
        
        response = await user_chat_sessions[user_id].send_message_async(
            request_content,
            safety_settings=safety_settings
        )
        
        final_text = clean_text_formatting(response.text)
        
        await send_long_message(update, final_text)
        
    except BlockedPromptException as e:
        logger.warning(f"Blocked content detected for user {user_id}: {e}")
        await update.message.reply_text(
            "Sorry, your request or response contains content that was blocked due to safety settings."
        )
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
        
        if user_id not in user_chat_sessions or user_chat_sessions[user_id].model != vision_audio_model:
            user_chat_sessions[user_id] = vision_audio_model.start_chat(history=INITIAL_HTML_INSTRUCTION)
            logger.info(f"Started new vision/audio chat session for user {user_id}.")
        
        prompt_for_gemini = "Transcribe the following voice message, and then respond to its content."
        
        response = await user_chat_sessions[user_id].send_message_async(
            [uploaded_file, prompt_for_gemini],
            safety_settings=safety_settings
        )
        
        final_text = clean_text_formatting(response.text)
        
        await send_long_message(update, final_text)
        
    except BlockedPromptException as e:
        logger.warning(f"Blocked content detected for user {user_id}: {e}")
        await update.message.reply_text(
            "Sorry, your request or response contains content that was blocked due to safety settings."
        )
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
            except Exception as e:
                logger.error(f"Error deleting Gemini file {uploaded_file.name}: {e}")

async def unhandled_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.message.from_user.id
    content_type = (update.message.effective_attachment.content_type
                    if update.message.effective_attachment else "unknown")
    logger.info(f"User {user_id} sent unhandled message type: {content_type}")
    
    spam_check_content = f"unhandled__{content_type}"
    if await check_spam(update, spam_check_content):
        return
    
    reply_text = (
        f"Sorry, I cannot process messages of type '{content_type}' yet.\n\n"
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
    
    try:
        application.run_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True
        )
    except Exception as e:
        logger.error(f"Bot stopped with error: {e}")
    finally:
        logger.info("Bot stopped.")

if __name__ == '__main__':
    main()
