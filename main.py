import argparse
import re
import yaml
import os
import sys
import time
import logging
from typing import List, Dict, Any, Optional, Callable

# External Libraries (ensure these are in requirements.txt)
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup, NavigableString
from openai import OpenAI, APIError, RateLimitError, AuthenticationError
import tiktoken
from tqdm import tqdm

# Attempt to import NLTK, but make it optional for basic sentence splitting
try:
    import nltk
    nltk.download('punkt', quiet=True) # Ensure 'punkt' tokenizer is available
    SENTENCE_TOKENIZER_AVAILABLE = True
except ImportError:
    nltk = None
    SENTENCE_TOKENIZER_AVAILABLE = False
    logging.warning("NLTK library not found. Falling back to basic regex-based sentence splitting. "
                    "For better sentence tokenization, please install NLTK: pip install nltk")

# --- Configuration ---
DEFAULT_CONFIG = {
    "openai": {
        "api_key": None, # Placeholder, should be set via ENV or actual config file
        "model": "gpt-4.1-mini",
        "temperature": 0.4,
        "max_tokens_per_chunk": 16384, # Based on GPT-4.1-mini's context is 32768. Leaving room for prompt and completion
        "request_timeout": 120, # Seconds
    },
    "translation_settings": {
        "retries": 3,
        "retry_delay_seconds": 10
    },
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(levelname)s - %(message)s"
    }
}

# --- Logging Setup ---
# Will be configured properly in main() after reading config
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def setup_logging(config: Dict[str, Any]) -> None:
    """Sets up global logging based on configuration."""
    log_config = config.get("logging", DEFAULT_CONFIG["logging"])
    level = getattr(logging, log_config.get("level", "INFO").upper(), logging.INFO)
    log_format = log_config.get("format", "%(asctime)s - %(levelname)s - %(message)s")
    logging.basicConfig(level=level, format=log_format, stream=sys.stdout)

def read_config(config_file: str) -> Dict[str, Any]:
    """
    Reads a YAML configuration file, merges with defaults, and returns its contents.
    """
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            user_config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.warning(f"Config file '{config_file}' not found. Using default settings.")
        return DEFAULT_CONFIG
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML config file '{config_file}': {e}. Using default settings.")
        return DEFAULT_CONFIG

    # Basic merge (user_config overrides defaults) - a more sophisticated merge might be needed for nested dicts
    config = DEFAULT_CONFIG.copy()
    if user_config: # Ensure user_config is not None
        for key, value in user_config.items():
            if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                config[key].update(value)
            else:
                config[key] = value
    return config

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Counts the number of tokens in a text string using tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        logger.warning(f"Model {model} not found for tiktoken. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def _nltk_sentence_tokenize(text: str, language: str = 'english') -> List[str]:
    """Tokenizes text into sentences using NLTK."""
    if not nltk or not SENTENCE_TOKENIZER_AVAILABLE:
        raise RuntimeError("NLTK sentence tokenizer called but NLTK is not available.")
    try:
        return nltk.tokenize.sent_tokenize(text, language=language)
    except LookupError: # pragma: no cover (in case punkt wasn't downloaded despite attempt)
        logger.warning(f"NLTK 'punkt' resource for language '{language}' not found. Falling back to default English.")
        nltk.download('punkt', quiet=True) # Re-attempt download
        return nltk.tokenize.sent_tokenize(text, language='english')
    except Exception as e:
        logger.error(f"Error during NLTK sentence tokenization: {e}. Falling back to regex.")
        return _regex_sentence_tokenize(text)


def _regex_sentence_tokenize(text: str) -> List[str]:
    """
    Basic regex-based sentence splitter. Less accurate than NLTK.
    Splits by '.', '!', '?' followed by a space or newline, trying to avoid abbreviations.
    """
    # More robust regex to handle various sentence endings and avoid splitting mid-sentence (e.g., Mr. Smith)
    # This is still a heuristic and not perfect.
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.|\?|!)\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def get_sentence_tokenizer(language_code: str) -> Callable[[str], List[str]]:
    """
    Returns the best available sentence tokenizer.
    Tries NLTK first for the specified language, then English NLTK, then regex.
    """
    if nltk and SENTENCE_TOKENIZER_AVAILABLE:
        # Map common language codes to NLTK language names if necessary
        # NLTK supports many, but names might differ (e.g., 'pt' -> 'portuguese')
        # For simplicity, we'll try common ones or default to English.
        # A more robust mapping might be needed for broader language support.
        lang_map = {
            'en': 'english', 'es': 'spanish', 'fr': 'french', 'de': 'german',
            'it': 'italian', 'pt': 'portuguese', 'nl': 'dutch', 'ru': 'russian'
            # Add more mappings as needed
        }
        nltk_lang = lang_map.get(language_code.lower(), 'english')
        try:
            # Test if the language is supported by NLTK's punkt
            nltk.tokenize.sent_tokenize("Test.", language=nltk_lang)
            logger.info(f"Using NLTK sentence tokenizer for language: {nltk_lang}")
            return lambda text: _nltk_sentence_tokenize(text, language=nltk_lang)
        except Exception: # pragma: no cover
            logger.warning(f"NLTK tokenizer for '{nltk_lang}' not fully available. Defaulting to English NLTK.")
            return lambda text: _nltk_sentence_tokenize(text, language='english')
    logger.info("Using regex-based sentence tokenizer.")
    return _regex_sentence_tokenize


def split_plain_text_into_chunks(text: str, max_tokens: int, model_name: str, from_lang_code: str) -> List[str]:
    """
    Splits plain text into chunks by sentences, keeping each chunk below max_tokens.
    """
    if not text.strip():
        return []

    sentence_tokenizer = get_sentence_tokenizer(from_lang_code)
    sentences = sentence_tokenizer(text)
    
    chunks = []
    current_chunk_sentences = []
    current_chunk_tokens = 0

    for sentence in sentences:
        sentence_tokens = count_tokens(sentence, model_name)
        
        if sentence_tokens > max_tokens:
            # If a single sentence is too long, split it further (simple split for now)
            # A more sophisticated approach might be needed for very long sentences
            logger.warning(f"A single sentence exceeds max_tokens ({sentence_tokens}/{max_tokens}). Splitting it: '{sentence[:50]}...'")
            # Simple split by words to fit, this might break meaning but is a fallback
            words = sentence.split()
            temp_sentence_part = ""
            for word in words:
                temp_sentence_part_with_word = temp_sentence_part + (" " if temp_sentence_part else "") + word
                if count_tokens(temp_sentence_part_with_word, model_name) <= max_tokens:
                    temp_sentence_part = temp_sentence_part_with_word
                else:
                    if temp_sentence_part: # Add the almost full part
                         chunks.append(temp_sentence_part)
                    temp_sentence_part = word # Start new part with current word
                    if count_tokens(temp_sentence_part, model_name) > max_tokens: # Single word too long
                        logger.error(f"A single word is too long for token limit: '{word[:50]}...'. Skipping this word.")
                        temp_sentence_part = "" # Skip it
            if temp_sentence_part:
                chunks.append(temp_sentence_part)
            # Reset current chunk as this long sentence was handled
            current_chunk_sentences = []
            current_chunk_tokens = 0
            continue

        if current_chunk_tokens + sentence_tokens <= max_tokens:
            current_chunk_sentences.append(sentence)
            current_chunk_tokens += sentence_tokens
        else:
            if current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))
            current_chunk_sentences = [sentence]
            current_chunk_tokens = sentence_tokens

    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))
        
    return [chunk for chunk in chunks if chunk.strip()]


def system_prompt(from_lang: str, to_lang: str) -> str:
    """Generates a system prompt for the translation task."""
    return (
        f"You are an expert {from_lang}-to-{to_lang} translator. "
        f"Translate the given text accurately and naturally. "
        f"Preserve the original meaning, tone, and context. "
        f"Maintain consistency with the source text. "
        f"Your translation should be in {to_lang} only. "
        f"IMPORTANT: If the input text contains HTML tags (like <b>, <i>, <span>, etc.), "
        f"reproduce these tags exactly as they are in the translated output, "
        f"wrapping the corresponding translated text. Do not translate the content of HTML attributes "
        f"like 'class' or 'id'. Only translate the text content."
    )

def translate_single_text_chunk(
    client: OpenAI, 
    text_chunk: str, 
    from_lang: str, 
    to_lang: str, 
    config: Dict[str, Any]
) -> Optional[str]:
    """Translates a single text chunk using OpenAI API with retries."""
    openai_config = config.get("openai", DEFAULT_CONFIG["openai"])
    translation_config = config.get("translation_settings", DEFAULT_CONFIG["translation_settings"])
    
    for attempt in range(translation_config["retries"]):
        try:
            response = client.chat.completions.create(
                model=openai_config["model"],
                temperature=openai_config["temperature"],
                messages=[
                    {'role': 'system', 'content': system_prompt(from_lang, to_lang)},
                    {'role': 'user', 'content': text_chunk},
                ],
                timeout=openai_config.get("request_timeout", 60)
            )
            translated_text = response.choices[0].message.content
            if translated_text:
                return translated_text.strip()
            else: # pragma: no cover
                logger.warning(f"Received empty translation for chunk: '{text_chunk[:50]}...'")
                return "" # Return empty string for empty translation
        except RateLimitError as e:
            logger.warning(f"Rate limit exceeded (attempt {attempt + 1}/{translation_config['retries']}). Retrying in {translation_config['retry_delay_seconds']}s... Error: {e}")
            time.sleep(translation_config['retry_delay_seconds'] * (attempt + 1)) # Exponential backoff could be better
        except APIError as e:
            logger.error(f"OpenAI API error (attempt {attempt + 1}/{translation_config['retries']}): {e}. Retrying...")
            time.sleep(translation_config['retry_delay_seconds'])
        except AuthenticationError as e: # pragma: no cover
            logger.error(f"OpenAI Authentication Error: {e}. Please check your API key and organization.")
            return None # Fatal error for this chunk
        except Exception as e: # pragma: no cover
            logger.error(f"An unexpected error occurred during translation (attempt {attempt + 1}/{translation_config['retries']}): {e}")
            time.sleep(translation_config['retry_delay_seconds'])
    
    logger.error(f"Failed to translate chunk after {translation_config['retries']} retries: '{text_chunk[:50]}...'")
    return None # Indicate failure

def process_and_translate_text_content(
    plain_text: str, 
    client: OpenAI, 
    from_lang: str, 
    to_lang: str, 
    config: Dict[str, Any]
) -> str:
    """Splits plain text, translates chunks, and reassembles them."""
    if not plain_text.strip():
        return plain_text # Return original if empty or whitespace only

    openai_config = config.get("openai", DEFAULT_CONFIG["openai"])
    
    chunks = split_plain_text_into_chunks(
        plain_text,
        openai_config["max_tokens_per_chunk"],
        openai_config["model"],
        from_lang # Pass from_lang for sentence tokenizer selection
    )

    if not chunks: # pragma: no cover
        logger.debug(f"No translatable chunks found for text: '{plain_text[:50]}...'")
        return plain_text # Return original if no chunks generated (e.g., all whitespace)

    translated_chunks = []
    # Use tqdm for chunk progress if there are multiple chunks
    chunk_iterator = tqdm(chunks, desc="Translating text chunks", unit="chunk", leave=False) if len(chunks) > 1 else chunks

    for i, chunk in enumerate(chunk_iterator):
        # logger.debug(f"Translating text chunk {i+1}/{len(chunks)} (tokens: {count_tokens(chunk, openai_config['model'])}): '{chunk[:70]}...'")
        translated_chunk = translate_single_text_chunk(client, chunk, from_lang, to_lang, config)
        if translated_chunk is not None:
            translated_chunks.append(translated_chunk)
        else:
            # If a chunk fails, append original to avoid data loss, but log error
            logger.error(f"Failed to translate chunk {i+1}, using original: '{chunk[:50]}...'")
            translated_chunks.append(chunk) # Fallback to original chunk

    # Join translated sentences/chunks. Usually a space is fine.
    # If sentence tokenization was perfect, this should be okay.
    return " ".join(translated_chunks)


def translate_html_element_content(
    element: BeautifulSoup, 
    client: OpenAI, 
    from_lang: str, 
    to_lang: str, 
    config: Dict[str, Any]
) -> None:
    """
    Recursively traverses a BeautifulSoup element, translates text nodes,
    and preserves HTML structure.
    """
    # Elements to skip translation for their content entirely
    # Add 'code' if you don't want code blocks translated, 'pre' for preformatted text
    block_skip_tags = ['script', 'style', 'pre', 'code'] 
    
    if element.name in block_skip_tags:
        return

    # Iterate over a copy of contents for safe modification
    for i, content_item in enumerate(list(element.contents)):
        if isinstance(content_item, NavigableString):
            text_to_translate = str(content_item).strip()
            if text_to_translate:
                # logger.debug(f"Found text node in <{element.name}>: '{text_to_translate[:50]}...'")
                translated_text = process_and_translate_text_content(
                    text_to_translate, client, from_lang, to_lang, config
                )
                content_item.replace_with(BeautifulSoup(translated_text, 'html.parser').contents[0] if translated_text else "")
        elif isinstance(content_item, BeautifulSoup) and content_item.name: # It's a tag
            # Recursively translate child elements
            translate_html_element_content(content_item, client, from_lang, to_lang, config)
        # Else: it might be a CData, Comment, etc. which we ignore for now.


def translate_epub(
    client: OpenAI, 
    input_epub_path: str, 
    output_epub_path: str, 
    from_lang: str, 
    to_lang: str,
    config: Dict[str, Any],
    from_chapter_num: int = 1, 
    to_chapter_num: int = float('inf')
) -> None:
    """
    Translates content of an EPUB book and writes the output to a new file.
    """
    try:
        book = epub.read_epub(input_epub_path)
    except FileNotFoundError: # pragma: no cover
        logger.error(f"Input EPUB file not found: {input_epub_path}")
        sys.exit(1)
    except Exception as e: # pragma: no cover
        logger.error(f"Error reading EPUB file '{input_epub_path}': {e}")
        sys.exit(1)

    items_to_translate = [item for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT)]
    total_chapters = len(items_to_translate)
    
    logger.info(f"Found {total_chapters} document items (chapters/sections) in the EPUB.")

    processed_chapter_count = 0
    for current_chapter_idx, item in enumerate(items_to_translate):
        # User-facing chapter numbers are 1-based
        current_chapter_user_num = current_chapter_idx + 1

        if not (from_chapter_num <= current_chapter_user_num <= to_chapter_num):
            # logger.info(f"Skipping chapter {current_chapter_user_num}/{total_chapters} (outside specified range).")
            continue
        
        processed_chapter_count += 1
        logger.info(f"Processing chapter {current_chapter_user_num}/{total_chapters} ('{item.get_name()}')...")
        
        try:
            original_html_content = item.get_content().decode('utf-8', errors='replace')
            soup = BeautifulSoup(original_html_content, 'html.parser')

            # Translate the body content, or the whole soup if no body (e.g. chapter is just a div)
            target_element_to_translate = soup.body if soup.body else soup
            if target_element_to_translate:
                translate_html_element_content(target_element_to_translate, client, from_lang, to_lang, config)
                # Update item content
                # Use prettify for better human readability of debug output, but str(soup) is fine for EPUB
                # item.content = soup.prettify(encoding='utf-8') 
                item.set_content(str(soup).encode('utf-8'))
            else: # pragma: no cover
                logger.warning(f"Chapter {current_chapter_user_num} ('{item.get_name()}') has no 'body' tag or content to translate.")

        except Exception as e: # pragma: no cover
            logger.error(f"Error processing chapter {current_chapter_user_num} ('{item.get_name()}'): {e}")
            # Decide if you want to skip this chapter or halt. For now, continue.
            logger.warning(f"Skipping translation for chapter {current_chapter_user_num} due to error.")
            continue # Skip to next item

    if processed_chapter_count == 0:
        logger.warning("No chapters were processed. Check your --from-chapter and --to-chapter range.")
    else:
        logger.info(f"Processed {processed_chapter_count} chapters.")

    try:
        epub.write_epub(output_epub_path, book, {})
        logger.info(f"Translated EPUB successfully saved to: {output_epub_path}")
    except Exception as e: # pragma: no cover
        logger.error(f"Error writing translated EPUB to '{output_epub_path}': {e}")
        sys.exit(1)


def show_chapters_info(input_epub_path: str) -> None:
    """Displays information about each chapter/document item in an EPUB book."""
    try:
        book = epub.read_epub(input_epub_path)
    except FileNotFoundError: # pragma: no cover
        logger.error(f"Input EPUB file not found: {input_epub_path}")
        sys.exit(1)
    except Exception as e: # pragma: no cover
        logger.error(f"Error reading EPUB file '{input_epub_path}': {e}")
        sys.exit(1)

    logger.info(f"EPUB Book Title: {book.get_metadata('DC', 'title')}")
    logger.info(f"EPUB Book Language: {book.get_metadata('DC', 'language')}")
    
    logger.info("\nTable of Contents (from NCX/Nav metadata if available):")
    if book.toc:
        for toc_item in book.toc:
            # toc_item can be a Link or a tuple (if nested)
            if isinstance(toc_item, epub.Link):
                logger.info(f"  - {toc_item.title} (href: {toc_item.href})")
            elif isinstance(toc_item, tuple) and len(toc_item) > 0 and isinstance(toc_item[0], epub.Link):
                 logger.info(f"  - {toc_item[0].title} (href: {toc_item[0].href}) (has sub-items)")

    logger.info("\nDocument Items (Chapters/Sections):")
    doc_items = [item for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT)]
    total_docs = len(doc_items)

    if not doc_items:
        logger.info("No document items found in this EPUB.")
        return

    for i, item in enumerate(doc_items):
        chapter_num = i + 1
        content_preview = "Error decoding content"
        char_count = 0
        try:
            content = item.get_content()
            char_count = len(content)
            soup = BeautifulSoup(content, 'html.parser')
            text_content = soup.get_text(separator=' ', strip=True)
            # Remove multiple newlines/spaces for cleaner preview
            text_content_cleaned = re.sub(r'\s+', ' ', text_content).strip()
            content_preview = text_content_cleaned[:250] + ('...' if len(text_content_cleaned) > 250 else '')
        except Exception as e: # pragma: no cover
            logger.debug(f"Could not parse chapter {chapter_num} for preview: {e}")
            content_preview = "[Content preview not available or unparseable]"

        logger.info(f"\n▶️  Chapter {chapter_num}/{total_docs}")
        logger.info(f"   Item Name: {item.get_name()}")
        logger.info(f"   File Name: {item.file_name}")
        logger.info(f"   Size (bytes): {char_count}")
        logger.info(f"   Preview: {content_preview}")

# --- Main Execution ---
def main():
    parser = argparse.ArgumentParser(description='Translate EPUB books using AI or show chapter information.')
    parser.add_argument('--config', default='config.yaml', help='Path to YAML configuration file (default: config.yaml).')
    
    subparsers = parser.add_subparsers(dest='mode', help='Mode of operation.', required=True)

    # Translate mode
    parser_translate = subparsers.add_parser('translate', help='Translate an EPUB book.')
    parser_translate.add_argument('--input', required=True, help='Input EPUB file path.')
    parser_translate.add_argument('--output', required=True, help='Output EPUB file path.')
    parser_translate.add_argument('--from-lang', required=True, help='Source language code (e.g., en, es, de).')
    parser_translate.add_argument('--to-lang', required=True, help='Target language code (e.g., en, es, de).')
    parser_translate.add_argument('--from-chapter', type=int, default=1, help='Starting chapter number (1-based) for translation (inclusive).')
    parser_translate.add_argument('--to-chapter', type=int, default=float('inf'), help='Ending chapter number (1-based) for translation (inclusive). Default: all chapters.')

    # Show chapters mode
    parser_show = subparsers.add_parser('show-chapters', help='Show chapter information from an EPUB file.')
    parser_show.add_argument('--input', required=True, help='Input EPUB file path.')

    args = parser.parse_args()

    # Read and merge configuration
    config = read_config(args.config)
    setup_logging(config) # Setup logging ASAP

    # API Key: Prioritize environment variable, then config file
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        api_key = config.get("openai", {}).get("api_key")
    
    if args.mode == 'translate':
        if not api_key:
            logger.error("OpenAI API key not found. Set OPENAI_API_KEY environment variable or add to config.yaml.")
            sys.exit(1)
        
        try:
            openai_client = OpenAI(api_key=api_key)
            # Test API connection (optional, but good for early failure)
            # openai_client.models.list() 
            logger.info("OpenAI client initialized.")
        except Exception as e: # pragma: no cover
            logger.error(f"Failed to initialize OpenAI client: {e}")
            sys.exit(1)

        logger.info(f"Starting EPUB translation from '{args.input}' to '{args.output}'.")
        logger.info(f"Languages: {args.from_lang} -> {args.to_lang}")
        logger.info(f"Chapter range: {args.from_chapter} to {'all' if args.to_chapter == float('inf') else args.to_chapter}")
        
        translate_epub(
            openai_client, 
            args.input, 
            args.output, 
            args.from_lang, 
            args.to_lang,
            config,
            args.from_chapter,
            args.to_chapter
        )

    elif args.mode == 'show-chapters':
        logger.info(f"Showing chapter information for: {args.input}")
        show_chapters_info(args.input)

    else: # Should not happen due to `required=True` on subparsers
        parser.print_help() # pragma: no cover

if __name__ == "__main__":
    main()
