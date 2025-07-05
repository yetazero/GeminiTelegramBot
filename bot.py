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
from dotenv import load_dotenv
from telegram.constants import ChatAction
from datetime import datetime, timedelta
import threading
import sys
import json

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ADMIN_USER_ID = os.getenv("ADMIN_USER_ID")

if not TELEGRAM_BOT_TOKEN or not GEMINI_API_KEY or not ADMIN_USER_ID:
    raise ValueError("TELEGRAM_BOT_TOKEN, GEMINI_API_KEY, and ADMIN_USER_ID environment variables must be set. Please check your .env file.")

ADMIN_USER_ID = int(ADMIN_USER_ID)

genai.configure(api_key=GEMINI_API_KEY)

CHAT_HISTORY_DIR = 'chat_histories'
FULL_CHAT_HISTORY_DIR = 'full_chat_histories'
FULL_HISTORY_USERS_FILE = 'full_history_users.json'

os.makedirs(CHAT_HISTORY_DIR, exist_ok=True)
os.makedirs(FULL_CHAT_HISTORY_DIR, exist_ok=True)

MAX_MESSAGES_TO_REMEMBER = 10
MAX_FULL_HISTORY_MESSAGES_TO_REMEMBER = 200

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
user_last_active = {}
SESSION_EXPIRATION_TIME = timedelta(hours=1)

RATE_LIMIT_SECONDS = 10
RATE_LIMIT_MESSAGES = 1
COOLDOWN_SECONDS = 30
REPEAT_MESSAGE_THRESHOLD = 2
user_activity = {}

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def get_history_file_path(user_id: int) -> str:
    return os.path.join(CHAT_HISTORY_DIR, f'{user_id}_history.json')

def get_full_history_file_path(user_id: int) -> str:
    return os.path.join(FULL_CHAT_HISTORY_DIR, f'{user_id}_full_history.json')

def load_full_history_users() -> set:
    if os.path.exists(FULL_HISTORY_USERS_FILE):
        try:
            with open(FULL_HISTORY_USERS_FILE, 'r', encoding='utf-8') as f:
                return set(json.load(f))
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding full_history_users.json: {e}. Starting with empty set.")
            return set()
    return set()

def save_full_history_users(users: set):
    with open(FULL_HISTORY_USERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(list(users), f, ensure_ascii=False, indent=2)

FULL_HISTORY_ENABLED_USERS = load_full_history_users()

def load_chat_history(user_id: int) -> list:
    file_path = get_history_file_path(user_id)
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON for user {user_id}: {e}. Starting new history.")
            return []
    return []

def save_chat_history(user_id: int, history: list):
    file_path = get_history_file_path(user_id)
    if len(history) > MAX_MESSAGES_TO_REMEMBER * 2:
        history = history[-MAX_MESSAGES_TO_REMEMBER * 2:]
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def load_full_chat_history(user_id: int) -> list:
    file_path = get_full_history_file_path(user_id)
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding full history JSON for user {user_id}: {e}. Starting new full history.")
            return []
    return []

def save_full_chat_history(user_id: int, history: list):
    file_path = get_full_history_file_path(user_id)
    if len(history) > MAX_FULL_HISTORY_MESSAGES_TO_REMEMBER:
        history = history[-MAX_FULL_HISTORY_MESSAGES_TO_REMEMBER:]
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

def clean_text_for_telegram(text: str) -> str:
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)
    text = re.sub(r'```[\s\S]*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'`(.*?)`', r'\1', text)
    text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^\s*[\*\-]\s', '', text, flags=re.MULTILINE)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = text.strip()
    return text

async def cleanup_old_sessions():
    now = datetime.now()
    expired_users = [
        user_id for user_id, last_active_time in user_last_active.items()
        if now - last_active_time > SESSION_EXPIRATION_TIME
    ]
    for user_id in expired_users:
        if user_id in user_chat_sessions:
            del user_chat_sessions[user_id]
        if user_id in user_last_active:
            del user_last_active[user_id]
        logger.info(f"Cleaned up in-memory session for user {user_id}")

async def send_long_message(update: Update, text: str):
    MAX_MESSAGE_LENGTH = 4096
    if len(text) <= MAX_MESSAGE_LENGTH:
        try:
            await update.message.reply_text(text)
        except Exception as e:
            logger.error(f"Failed to send message for user {update.message.from_user.id}: {e}")
        return

    parts = []
    current_part = ""
    lines = text.split('\n')
    for line in lines:
        if len(current_part) + len(line) + 1 > MAX_MESSAGE_LENGTH:
            if current_part:
                parts.append(current_part)
            current_part = line
            while len(current_part) > MAX_MESSAGE_LENGTH:
                parts.append(current_part[:MAX_MESSAGE_LENGTH])
                current_part = current_part[MAX_MESSAGE_LENGTH:]
        else:
            current_part += ('\n' if current_part else '') + line
    if current_part:
        parts.append(current_part)

    for part in parts:
        if part.strip():
            try:
                await update.message.reply_text(part)
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.error(f"Failed to send message part for user {update.message.from_user.id}: {e}")

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
        await update.message.reply_text(f"Please wait {remaining_time} seconds before sending a new request.")
        return True
    
    activity['timestamps'] = [t for t in activity['timestamps'] if current_time - t < RATE_LIMIT_SECONDS]
    activity['timestamps'].append(current_time)
    
    if (message_content and message_content.lower() == activity['last_message'].lower()):
        activity['last_message_count'] += 1
    else:
        activity['last_message'] = message_content
        activity['last_message_count'] = 1
        
    if activity['last_message_count'] > REPEAT_MESSAGE_THRESHOLD:
        await update.message.reply_text("I have already received this message. Please send something new.")
        return True
    
    if len(activity['timestamps']) > RATE_LIMIT_MESSAGES:
        activity['blocked_until'] = current_time + COOLDOWN_SECONDS
        await update.message.reply_text(f"You are sending too many messages. Please wait {COOLDOWN_SECONDS} seconds.")
        logger.warning(f"User {user_id} hit rate limit, blocked for {COOLDOWN_SECONDS} seconds.")
        return True
    
    return False

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info(f"User {update.message.from_user.id} started the bot.")
    await update.message.reply_text('Hi! I am a Gemini-based bot. Send me a message, photo, or voice message.')

async def clear_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.message.from_user.id
    if user_id in user_chat_sessions:
        del user_chat_sessions[user_id]
    if user_id in user_last_active:
        del user_last_active[user_id]

    std_file_path = get_history_file_path(user_id)
    if os.path.exists(std_file_path):
        os.remove(std_file_path)
        logger.info(f"Deleted standard chat history file for user {user_id}.")
    
    full_file_path = get_full_history_file_path(user_id)
    if os.path.exists(full_file_path):
        os.remove(full_file_path)
        logger.info(f"Deleted full chat history file for user {user_id}.")
    
    logger.info(f"All chat history cleared for user {user_id}.")
    await update.message.reply_text("All chat history cleared for you. We can start a new conversation now.")

async def history_control(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    global FULL_HISTORY_ENABLED_USERS
    
    if update.message.from_user.id != ADMIN_USER_ID:
        await update.message.reply_text("You are not authorized to use this command.")
        logger.warning(f"Unauthorized access to /history command by user {update.message.from_user.id}")
        return

    args = context.args
    if len(args) != 2 or args[0].lower() not in ["on", "off"]:
        await update.message.reply_text("Usage: /history <on|off> <user_id>")
        return

    action = args[0].lower()
    try:
        target_user_id = int(args[1])
    except ValueError:
        await update.message.reply_text("Invalid user ID. Please provide a numeric ID.")
        return

    if action == "on":
        FULL_HISTORY_ENABLED_USERS.add(target_user_id)
        save_full_history_users(FULL_HISTORY_ENABLED_USERS)
        await update.message.reply_text(f"Full history logging ENABLED for user {target_user_id}.")
        logger.info(f"Admin {ADMIN_USER_ID} enabled full history for user {target_user_id}")
    elif action == "off":
        if target_user_id in FULL_HISTORY_ENABLED_USERS:
            FULL_HISTORY_ENABLED_USERS.remove(target_user_id)
            save_full_history_users(FULL_HISTORY_ENABLED_USERS)
            await update.message.reply_text(f"Full history logging DISABLED for user {target_user_id}.")
            logger.info(f"Admin {ADMIN_USER_ID} disabled full history for user {target_user_id}")
        else:
            await update.message.reply_text(f"Full history logging was not enabled for user {target_user_id}.")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.message.from_user.id
    user_message = update.message.text
    logger.info(f"User {user_id} sent text message: {user_message}")
    
    if await check_spam(update, user_message):
        return
        
    await cleanup_old_sessions()
    
    await update.message.chat.send_action(ChatAction.TYPING)
    
    current_history = []
    if user_id in FULL_HISTORY_ENABLED_USERS:
        current_history = load_full_chat_history(user_id)
        logger.info(f"Loaded FULL chat history for user {user_id} ({len(current_history)} messages).")
    else:
        current_history = load_chat_history(user_id)
        logger.info(f"Loaded standard chat history for user {user_id} ({len(current_history)} messages).")

    gemini_history = INITIAL_HTML_INSTRUCTION + current_history

    try:
        chat_session = text_model.start_chat(history=gemini_history)

        response = await chat_session.send_message_async(user_message, safety_settings=safety_settings)
        final_text = clean_text_for_telegram(response.text)
        
        current_history.append({"role": "user", "parts": [user_message]})
        current_history.append({"role": "model", "parts": [response.text]})
        
        if user_id in FULL_HISTORY_ENABLED_USERS:
            save_full_chat_history(user_id, current_history)
        else:
            save_chat_history(user_id, current_history)
        
        user_last_active[user_id] = datetime.now()
        
        await send_long_message(update, final_text)
    except BlockedPromptException as e:
        logger.warning(f"Blocked content detected for user {user_id}: {e}")
        await update.message.reply_text("Sorry, your request contains content that was blocked due to safety settings.")
    except Exception as e:
        logger.error(f"Error processing text message from {user_id}: {e}", exc_info=True)
        await update.message.reply_text(f"An error occurred while processing your request: {e}")

async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.message.from_user.id
    caption_prompt = update.message.caption or ""
    logger.info(f"User {user_id} sent photo with caption: {caption_prompt}")
    
    spam_check_content = caption_prompt if caption_prompt else f"photo__{update.message.photo[-1].file_id}"
    if await check_spam(update, spam_check_content):
        return
        
    await cleanup_old_sessions()
    
    await update.message.reply_text("Received image, processing...")
    await update.message.chat.send_action(ChatAction.UPLOAD_PHOTO)
    
    file_id = update.message.photo[-1].file_id
    file = await context.bot.get_file(file_id)
    
    uploaded_file = None
    
    current_history = []
    if user_id in FULL_HISTORY_ENABLED_USERS:
        current_history = load_full_chat_history(user_id)
        logger.info(f"Loaded FULL chat history for user {user_id} (vision/audio).")
    else:
        current_history = load_chat_history(user_id)
        logger.info(f"Loaded standard chat history for user {user_id} (vision/audio).")

    gemini_history = INITIAL_HTML_INSTRUCTION + current_history

    try:
        with BytesIO() as bio:
            await file.download_to_memory(bio)
            bio.seek(0)
            image = Image.open(bio)
            
            img_byte_arr = BytesIO()
            image.save(img_byte_arr, format='JPEG')
            img_byte_arr.seek(0)

            uploaded_file = genai.upload_file(path=img_byte_arr, display_name=f"{file_id}.jpg", mime_type="image/jpeg")
            logger.info(f"Photo uploaded to Gemini: {uploaded_file.name}")

        chat_session = vision_audio_model.start_chat(history=gemini_history)
            
        user_last_active[user_id] = datetime.now()
        
        request_content = [uploaded_file, caption_prompt] if caption_prompt else [uploaded_file]
        response = await chat_session.send_message_async(request_content, safety_settings=safety_settings)
        final_text = clean_text_for_telegram(response.text)
        
        current_history.append({"role": "user", "parts": [caption_prompt if caption_prompt else "User sent an image."]})
        current_history.append({"role": "model", "parts": [response.text]})
        
        if user_id in FULL_HISTORY_ENABLED_USERS:
            save_full_chat_history(user_id, current_history)
        else:
            save_chat_history(user_id, current_history)

        await send_long_message(update, final_text)
        
    except BlockedPromptException as e:
        logger.warning(f"Blocked content detected for user {user_id}: {e}")
        await update.message.reply_text("Sorry, your request contains content that was blocked due to safety settings.")
    except Exception as e:
        logger.error(f"Error processing photo from {user_id}: {e}", exc_info=True)
        await update.message.reply_text(f"An error occurred while processing your photo: {e}")
    finally:
        if uploaded_file and hasattr(uploaded_file, 'name'):
            try:
                await genai.delete_file_async(uploaded_file.name)
                logger.info(f"Deleted Gemini file {uploaded_file.name}")
            except Exception as e:
                logger.error(f"Error deleting Gemini file {uploaded_file.name}: {e}")

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.message.from_user.id
    logger.info(f"User {user_id} sent voice message.")
    
    spam_check_content = f"voice__{update.message.voice.file_id}"
    if await check_spam(update, spam_check_content):
        return

    await cleanup_old_sessions()

    await update.message.reply_text("Received voice message, processing...")
    await update.message.chat.send_action(ChatAction.UPLOAD_VOICE)
    
    file_id = update.message.voice.file_id
    file = await context.bot.get_file(file_id)
    
    uploaded_file = None

    current_history = []
    if user_id in FULL_HISTORY_ENABLED_USERS:
        current_history = load_full_chat_history(user_id)
        logger.info(f"Loaded FULL chat history for user {user_id} (voice).")
    else:
        current_history = load_chat_history(user_id)
        logger.info(f"Loaded standard chat history for user {user_id} (voice).")
    
    gemini_history = INITIAL_HTML_INSTRUCTION + current_history
    
    try:
        with BytesIO() as bio:
            await file.download_to_memory(bio)
            bio.seek(0)
            uploaded_file = genai.upload_file(path=bio, display_name=f"{file_id}.ogg", mime_type="audio/ogg")
            logger.info(f"Voice message uploaded to Gemini: {uploaded_file.name}")

        chat_session = vision_audio_model.start_chat(history=gemini_history)
            
        user_last_active[user_id] = datetime.now()
        
        prompt_for_gemini = "Transcribe the following voice message, and then respond to its content."
        response = await chat_session.send_message_async([uploaded_file, prompt_for_gemini], safety_settings=safety_settings)
        final_text = clean_text_for_telegram(response.text)

        current_history.append({"role": "user", "parts": [f"User sent a voice message. Context prompt: {prompt_for_gemini}"]})
        current_history.append({"role": "model", "parts": [response.text]})
        
        if user_id in FULL_HISTORY_ENABLED_USERS:
            save_full_chat_history(user_id, current_history)
        else:
            save_chat_history(user_id, current_history)

        await send_long_message(update, final_text)
        
    except BlockedPromptException as e:
        logger.warning(f"Blocked content detected for user {user_id}: {e}")
        await update.message.reply_text("Sorry, your request contains content that was blocked due to safety settings.")
    except Exception as e:
        logger.error(f"Error processing voice message from {user_id}: {e}", exc_info=True)
        await update.message.reply_text(f"An error occurred while processing your voice message: {e}")
    finally:
        if uploaded_file and hasattr(uploaded_file, 'name'):
            try:
                await genai.delete_file_async(uploaded_file.name)
                logger.info(f"Deleted Gemini file {uploaded_file.name}")
            except Exception as e:
                logger.error(f"Error deleting Gemini file {uploaded_file.name}: {e}")

async def unhandled_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.info(f"User {update.message.from_user.id} sent an unhandled message type.")
    await update.message.reply_text("Sorry, I can only process text messages, photos, and voice messages.")

def restart_bot():
    logger.info("Restarting bot...")
    python = sys.executable
    os.execl(python, python, *sys.argv)

def schedule_restart():
    threading.Timer(3600, restart_bot).start()

def main() -> None:
    application = (
        Application.builder()
        .token(TELEGRAM_BOT_TOKEN)
        .connect_timeout(30)
        .read_timeout(30)
        .write_timeout(30)
        .pool_timeout(60)
        .connection_pool_size(128)
        .build()
    )
    
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("clear", clear_history))
    application.add_handler(CommandHandler("history", history_control))
    
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice))
    application.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND, unhandled_message))
    
    logger.info("Bot started successfully.")
    
    schedule_restart()

    try:
        application.run_polling(allowed_updates=Update.ALL_TYPES)
    except Exception as e:
        logger.critical(f"Bot stopped with a critical error: {e}", exc_info=True)
    finally:
        logger.info("Bot stopped.")

if __name__ == '__main__':
    main()
