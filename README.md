# Gemini Telegram Bot

This is a Python-based Telegram bot that leverages Google's Gemini API to provide interactive text, image analysis, and voice message transcription capabilities. The bot is built using `python-telegram-bot` and `google-generativeai` libraries, offering a conversational experience with multi-modal input support.

---

## Features

* **Text Conversations:** Engage in natural language conversations powered by the Gemini 1.5 Flash model.
* **Image Understanding:** Send photos with or without captions, and the bot will analyze them and respond.
* **Voice Message Transcription:** Upload voice messages, and the bot will transcribe them and provide a relevant response.
* **HTML Formatting:** Responses from the bot are formatted using a subset of HTML tags supported by Telegram, ensuring readable and structured output.
* **Conversation History:** Maintains context within a chat session for a more coherent conversation flow.
* **Spam Protection:** Implements basic rate-limiting and repeat message detection to prevent abuse.
* **Error Handling:** Gracefully handles API errors and content blocking exceptions from Gemini.

---

## Getting Started

Follow these steps to set up and run your Gemini Telegram Bot.

### Prerequisites

* Python 3.9+
* A Telegram Bot Token (obtain from BotFather on Telegram)
* A Google Cloud Project with the Gemini API enabled and an API Key

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/yetazero/GeminiTelegramBot](https://github.com/yetazero/GeminiTelegramBot)
    cd YOUR_REPO_NAME
    ```
2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    (Create a `requirements.txt` file if you don't have one, containing:
    ```
    python-telegram-bot
    google-generativeai
    python-dotenv
    Pillow
    ```
    )

### Configuration

1.  **Create a `.env` file** in the root directory of your project:
    ```
    TELEGRAM_BOT_TOKEN="YOUR_TELEGRAM_BOT_TOKEN"
    GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
    ```
    Replace `"YOUR_TELEGRAM_BOT_TOKEN"` with your actual Telegram bot token and `"YOUR_GEMINI_API_KEY"` with your Google Gemini API key.

### Running the Bot

```bash
python your_bot_file_name.py
(Replace your_bot_file_name.py with the actual name of your Python script, e.g., main.py if you named it that).

Usage
Send /start to begin interacting with the bot.
Use /clear to reset the conversation history.
Send a text message to chat with the Gemini model.
Send a photo (with or without a caption) for image analysis.
Send a voice message for transcription and a response based on its content.
Contributing
Feel free to fork the repository, open issues, or submit pull requests.

License
This project is open-source and available under the MIT License.
