# 🌎 This repository is a fork of jb41/translate-book

# Translate books with GPT

This project harnesses the power of Large Language Models (LLMs) like GPT-4.1-mini to translate eBooks from any language into your preferred language, maintaining the integrity and structure of the original content. Imagine having access to a vast world of literature, regardless of the original language, right at your fingertips.

This tool not only translates the text but also carefully reassembles each element of the eBook – chapters, footnotes, and all – into a perfectly formatted EPUB file. We use `gpt-4.1-mini` by default for a good balance of quality and cost, but you can easily change the model and other settings in the `config.yaml` file.

## 💵 Cost

The cost of translations using models like `gpt-4.1-mini` is generally quite low. For indicative pricing (please always check OpenAI's official pricing page for the latest rates, for example, as of mid-2025 for `gpt-4.1-mini`):

```
Input: ~$0.50 per 1 million tokens
Output: ~$1.60 per 1 million tokens
```

An average web novel chapter might contain about 2500 tokens (around 9000 characters).
**This means translating a book with 400 such chapters could cost you less than $3 USD.** (Calculation: 400 chapters * 2500 tokens/chapter = 1M input tokens. Assuming output is similar, total cost would be around $0.50 + $1.60 = $2.10).

## 🛠️ Installation

Follow these steps to get the translator up and running on your computer:

1.  **Create a Project Folder:**
    Create a new folder on your machine where you want to store this project. For example, in your "Documents" folder, you could create "novel-translation".

2.  **Open a Terminal (Command Prompt/PowerShell/Terminal):**
    *   **Windows:** Search for "cmd" or "PowerShell".
    *   **macOS/Linux:** Search for "Terminal".
    Navigate to the folder you just created. If your folder path is `C:\Users\YourName\Documents\epub-translation`, you would type:
    ```bash
    cd "C:\Users\YourName\Documents\epub-translation"
    ```
    *(Remember the quotes if your path has spaces!)*

3.  **Clone the Repository:**
    In the terminal, run this command to download the project files:
    ```bash
    git clone https://github.com/Odysseum04/epub-gpt-translation.git
    ```
    This will create a new folder named `epub-gpt-translation` inside your project directory.

4.  **Open the Project:**
    Open the `novel-gpt-translation` folder in your favorite code editor (like VS Code, PyCharm, Sublime Text, etc.).

5.  **Set up Configuration & API Key:**
    *   Inside the `epub-gpt-translation` folder, find the file named `config.yaml.example`.
    *   **Copy this file and rename the copy to `config.yaml`** in the same directory.
    *   **Open `config.yaml` with your editor.** You'll need to add your OpenAI API key here:
        ```yaml
        openai:
          api_key: "sk-YOUR_OPENAI_API_KEY_HERE" # Replace with your actual key
          # ... other settings ...
        ```
    *   **How to get an OpenAI API Key?** You can find many tutorials online, for example: [YouTube Tutorial Link (General Guide)](https://www.youtube.com/watch?v=nafDyRsVnXU)
    *   **Alternatively (Recommended & More Secure):** Instead of putting the key in `config.yaml`, you can set an environment variable named `OPENAI_API_KEY` with your key value. The script will automatically use it if set.

6.  **Create a Python Virtual Environment (Highly Recommended):**
    A virtual environment keeps project dependencies isolated. In your terminal (ensure you are inside the `novel-gpt-translation` folder):
    ```bash
    python -m venv .venv 
    ```
    Then, activate it:
    *   **Windows (cmd):** `.venv\Scripts\activate.bat`
    *   **Windows (PowerShell):** `.venv\Scripts\Activate.ps1` (You might need to allow script execution: `Set-ExecutionPolicy Unrestricted -Scope Process`)
    *   **macOS/Linux:** `source .venv/bin/activate`
    You should see `(.venv)` at the beginning of your terminal prompt.

7.  **Install Dependencies:**
    With your virtual environment activated, install the necessary Python packages:
    ```bash
    pip install -r requirements.txt
    ```
    This will install `ebooklib`, `beautifulsoup4`, `openai`, `PyYAML`, `tiktoken`, `tqdm`, and `nltk`. NLTK is used for better sentence splitting during translation.

You're all set up!

## 🎮 Usage

The script is run from the terminal, inside the `epub-gpt-translation` folder (make sure your virtual environment is activated).

### Show Chapters

Before translating, it's a good idea to see how the book is structured:
```bash
python main.py show-chapters --input yourbook.epub
```
This command lists the chapters (or document items) in your EPUB file, their approximate size, and a short preview. This helps you decide if you want to translate specific chapters.

### Translate Mode

To translate a book:
```bash
python main.py translate --input yourbook.epub --output translatedbook.epub --from-lang EN --to-lang FR --from-chapter 1 --to-chapter 5
```

**Explanation of options:**

*   `--input yourbook.epub`: Path to the EPUB file you want to translate.
*   `--output translatedbook.epub`: Path where the translated EPUB will be saved.
*   `--from-lang EN`: The language code of the original book (e.g., `EN` for English, `JA` for Japanese, `ZH` for Chinese).
*   `--to-lang FR`: The language code you want to translate to (e.g., `FR` for French, `ES` for Spanish, `DE` for German).
*   `--from-chapter 1` (Optional): The chapter number to start translating from (inclusive). Defaults to the first chapter.
*   `--to-chapter 5` (Optional): The chapter number to end translating at (inclusive). Defaults to the last chapter.
*   `--config config.yaml` (Optional): Path to your configuration file. If not specified, it defaults to `config.yaml` in the current directory.

The script will provide progress updates in the terminal as it processes and translates chapters and chunks of text.

### Advanced Configuration

You can fine-tune the translation process by editing the `config.yaml` file. Settings you can change include:
*   OpenAI model (e.g., `gpt-4.1-mini`, `gpt-4-turbo`)
*   Translation temperature (creativity vs. precision)
*   Maximum tokens per chunk sent to the API
*   Retry attempts for API calls
*   Logging level

## 🤝 Source
This repository is a fork of [jb41/translate-book](https://github.com/jb41/translate-book). Don't forget to thank the original author by starring their GitHub repo too! The current, reworked version of the script aims to be robust and user-friendly.
