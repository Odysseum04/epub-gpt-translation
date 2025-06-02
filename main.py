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
from bs4 import BeautifulSoup # NavigableString might not be directly used now
from openai import OpenAI, APIError, RateLimitError, AuthenticationError
import tiktoken
from tqdm import tqdm

# Attempt to import NLTK (kept for now, but not used in this translation strategy)
try:
    import nltk
    SENTENCE_TOKENIZER_AVAILABLE = True # Variable kept for consistency
except ImportError:
    nltk = None
    SENTENCE_TOKENIZER_AVAILABLE = False

# --- Configuration ---
DEFAULT_CONFIG = {
    "openai": {
        "api_key": None, 
        "model": "gpt-4.1-mini", 
        "temperature": 0.4,
        # CRITICAL FOR THIS STRATEGY: Increase this significantly in your config.yaml!
        # e.g., 16000, 32000, or higher, up to model limits minus prompt/output.
        "max_tokens_per_chunk": 16000, # Default is low for this strategy.
        "request_timeout": 1200, # Increased default timeout for potentially larger payloads
    },
    "translation_settings": {
        "retries": 3,
        "retry_delay_seconds": 15 # Increased delay for potentially larger retries
    },
    "logging": {
        "level": "INFO",
        "format": "%(asctime)s - %(levelname)s - %(message)s"
    }
}

# --- Logging Setup ---
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def setup_logging(config: Dict[str, Any]) -> None:
    """Sets up global logging and NLTK (though NLTK not used in this strategy)."""
    log_config = config.get("logging", DEFAULT_CONFIG["logging"])
    level_str = log_config.get("level", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    log_format = log_config.get("format", "%(asctime)s - %(levelname)s - %(message)s")
    logging.basicConfig(level=level, format=log_format, stream=sys.stdout, force=True)
    logger.info(f"Logging configured to level: {level_str}")

    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    if not SENTENCE_TOKENIZER_AVAILABLE:
        logger.debug("NLTK library not found (not critical for current large-block strategy).")
    elif nltk:
        logger.debug("NLTK library found (not actively used for sentence tokenization in current large-block strategy).")
        # NLTK punkt download attempt (kept for completeness, harmless if not used)
        # resources_to_download = ['punkt']
        # for resource_name in resources_to_download:
        #     try:
        #         nltk.data.find(f'tokenizers/{resource_name}')
        #         logger.debug(f"NLTK resource '{resource_name}' found.")
        #     except nltk.downloader.DownloadError: 
        #         logger.info(f"NLTK resource '{resource_name}' not found. Attempting download (quietly)...")
        #         try:
        #             nltk.download(resource_name, quiet=True) 
        #             nltk.data.find(f'tokenizers/{resource_name}')
        #             logger.info(f"NLTK resource '{resource_name}' is now available after download attempt.")
        #         except Exception: # Catch all for download/find issues
        #             logger.warning(f"Could not download/verify NLTK resource '{resource_name}'.")


def read_config(config_file: str) -> Dict[str, Any]:
    """Reads a YAML configuration file, merges with defaults, and returns its contents."""
    try:
        with open(config_file, 'r', encoding='utf-8') as f:
            user_config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.info(f"Config file '{config_file}' not found. Using default settings and environment variables.")
        return DEFAULT_CONFIG
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML config file '{config_file}': {e}. Using default settings.")
        return DEFAULT_CONFIG

    config = DEFAULT_CONFIG.copy()
    if user_config:
        for key, value in user_config.items():
            if key in config and isinstance(config[key], dict) and isinstance(value, dict):
                config[key].update(value)
            else:
                config[key] = value
    return config

def count_tokens(text: str, model: str) -> int:
    """Counts the number of tokens in a text string using tiktoken."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        logger.error(f"TIKTOKEN ERROR: Model '{model}' not found by tiktoken. "
                     f"Token counting will be inaccurate; API calls with this model name WILL LIKELY FAIL. "
                     f"Use a valid OpenAI model name (e.g., 'gpt-4o-mini'). "
                     f"Falling back to 'cl100k_base' for token counting (NOT a fix for API model name).")
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def system_prompt(from_lang: str, to_lang: str) -> str:
    """Generates a system prompt for the translation task."""
    return (
        f"You are an expert {from_lang}-to-{to_lang} translator. "
        f"Translate ONLY the human-readable text content within the provided HTML. "
        f"PRESERVE ALL HTML TAGS AND THEIR ATTRIBUTES EXACTLY AS THEY ARE IN THE ORIGINAL. "
        f"Do not add, remove, or modify any HTML tags or attributes. "
        f"Do not translate content of attributes like 'class', 'id', 'href', 'src'. "
        f"Ensure the output is valid HTML with translated text content. "
        f"Your translation should be in {to_lang} only. "
        f"Maintain the original meaning, tone, and context of the text."
    )

def translate_single_text_chunk( # Renamed from translate_single_text_chunk for clarity, now handles HTML blocks
    client: OpenAI, html_block_to_translate: str, from_lang: str, to_lang: str, config: Dict[str, Any]
) -> Optional[str]:
    """Translates a single block of (potentially HTML) text using OpenAI API with retries."""
    openai_config = config.get("openai", DEFAULT_CONFIG["openai"])
    translation_config = config.get("translation_settings", DEFAULT_CONFIG["translation_settings"])
    
    model_to_use = openai_config["model"]
    logger.debug(f"translate_single_html_block: Attempting to translate block using OpenAI model: '{model_to_use}'. Block (first 150 chars): '{html_block_to_translate[:150]}...'")

    for attempt in range(translation_config["retries"]):
        try:
            logger.debug(f"  API Call Attempt {attempt + 1}/{translation_config['retries']}")
            response = client.chat.completions.create(
                model=model_to_use,
                temperature=openai_config["temperature"],
                messages=[
                    {'role': 'system', 'content': system_prompt(from_lang, to_lang)},
                    {'role': 'user', 'content': html_block_to_translate}, # Sending the HTML block
                ],
                timeout=openai_config.get("request_timeout", DEFAULT_CONFIG["openai"]["request_timeout"])
            )
            translated_text = response.choices[0].message.content
            logger.debug(f"  API Call Success. Raw translated response (first 150 chars): '{translated_text[:150] if translated_text else 'EMPTY_RESPONSE'}'")
            if translated_text:
                # We expect the LLM to return valid HTML with translated text.
                # No .strip() here, as whitespace might be significant in HTML structure.
                return translated_text 
            else:
                logger.warning(f"Received empty translation from API for HTML block: '{html_block_to_translate[:70]}...'")
                return "" 
        except RateLimitError as e:
            logger.warning(f"Rate limit exceeded (attempt {attempt + 1}/{translation_config['retries']}). Retrying in {translation_config['retry_delay_seconds'] * (attempt + 1)}s... Error: {e}")
            time.sleep(translation_config['retry_delay_seconds'] * (attempt + 1))
        except APIError as e:
            if "does not exist" in str(e).lower() or "invalid model" in str(e).lower():
                logger.error(f"FATAL OpenAI API Error: Model '{model_to_use}' likely does not exist or you do not have access. "
                             f"Please check the model name in your configuration. Error: {e}", exc_info=True)
                return None 
            # Check for context length exceeded errors
            if "context_length_exceeded" in str(e).lower() or "maximum context length" in str(e).lower():
                logger.error(f"OpenAI API Error: Model '{model_to_use}' context length exceeded. "
                             f"The HTML block sent was too large for the model's input capacity. Error: {e}", exc_info=True)
                return None # Don't retry if context length is the issue for this block
            logger.error(f"OpenAI API error (attempt {attempt + 1}/{translation_config['retries']}): {e}. Retrying...", exc_info=True)
            time.sleep(translation_config['retry_delay_seconds'])
        except AuthenticationError as e: 
            logger.error(f"OpenAI Authentication Error: {e}. Please check your API key and organization. Halting translation for this block.", exc_info=True)
            return None 
        except Exception as e: 
            logger.error(f"An unexpected error occurred during translation (attempt {attempt + 1}/{translation_config['retries']}): {e}", exc_info=True)
            time.sleep(translation_config['retry_delay_seconds'])
    
    logger.error(f"Failed to translate HTML block after {translation_config['retries']} retries: '{html_block_to_translate[:70]}...'")
    return None

def split_html_into_large_blocks(html_content: str, max_tokens: int, model_name: str) -> List[str]:
    """
    Splits HTML content into blocks if it exceeds max_tokens.
    WARNING: Current implementation is naive and may break HTML.
    A robust HTML-aware splitter is needed for production if chapters frequently exceed token limits.
    """
    total_tokens = count_tokens(html_content, model_name)
    logger.debug(f"split_html_into_large_blocks: Total tokens for HTML content: {total_tokens}, max_tokens_per_chunk: {max_tokens}")

    if total_tokens <= max_tokens:
        logger.debug("split_html_into_large_blocks: Entire HTML content fits in one block.")
        return [html_content]

    logger.warning(
        f"HTML content (tokens: {total_tokens}) exceeds max_tokens_per_chunk ({max_tokens}). "
        "Attempting a NAIVE character-based split. THIS IS VERY LIKELY TO BREAK HTML STRUCTURE. "
        "The translated EPUB may be corrupted for this chapter. "
        "Strongly consider increasing 'max_tokens_per_chunk' in config.yaml or implementing a "
        "proper HTML-aware block splitting mechanism (e.g., splitting between major <div> or <p> tags)."
    )
    
    # Extremely naive character-based split as a last resort placeholder.
    # This does NOT respect HTML tags and will likely break them.
    num_chunks_needed = (total_tokens + max_tokens - 1) // max_tokens # Ceiling division for token count
    # Use character length for actual split, proportionally
    estimated_chars_per_token = len(html_content) / total_tokens if total_tokens > 0 else 4 # Rough estimate
    target_chars_per_chunk = int(max_tokens * estimated_chars_per_token * 0.95) # 0.95 for safety margin

    if target_chars_per_chunk <=0: # Safety for very short content but high token count (unlikely)
        target_chars_per_chunk = len(html_content) // num_chunks_needed if num_chunks_needed > 0 else len(html_content)


    chunks = []
    current_pos = 0
    while current_pos < len(html_content):
        end_pos = min(current_pos + target_chars_per_chunk, len(html_content))
        # Try to find a space to split on, to avoid breaking mid-word, but still crude for HTML
        # This is not a good way to split HTML.
        if end_pos < len(html_content):
            last_space = html_content.rfind(' ', current_pos, end_pos)
            if last_space != -1 and last_space > current_pos : # if space found and not at the beginning
                # A better check would be if this space is outside a tag. This is too simple.
                # For now, just split at char count.
                pass # Keeping it simple char split due to complexity of HTML-aware split here.

        chunks.append(html_content[current_pos:end_pos])
        current_pos = end_pos
    
    logger.debug(f"split_html_into_large_blocks: Naively split HTML into {len(chunks)} character-based blocks.")
    return chunks

def translate_html_chapter_content(
    html_to_translate: str, 
    client: OpenAI, 
    from_lang: str, 
    to_lang: str, 
    config: Dict[str, Any]
) -> str:
    """
    Manages splitting (if necessary) and translating the HTML content of a chapter.
    """
    openai_config = config.get("openai", DEFAULT_CONFIG["openai"])
    max_tokens_for_html_chunk = openai_config["max_tokens_per_chunk"]
    model_name = openai_config["model"]

    html_blocks = split_html_into_large_blocks(html_to_translate, max_tokens_for_html_chunk, model_name)
    
    translated_html_pieces = []
    num_blocks = len(html_blocks)

    # Only show block progress bar if there's more than one block AND we are not in deep debug mode
    show_block_pbar = (num_blocks > 1) and (logger.getEffectiveLevel() > logging.DEBUG)

    block_iterator = tqdm(
        html_blocks,
        desc="  ↪ Translating HTML blocks", 
        unit="block",
        leave=False, 
        disable=not show_block_pbar,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    )

    for i, block in enumerate(block_iterator):
        if not show_block_pbar and logger.getEffectiveLevel() <= logging.DEBUG:
            logger.debug(f"    Translating HTML block {i+1}/{num_blocks}")
        
        # Use the renamed translate_single_text_chunk which now expects HTML blocks
        translated_piece = translate_single_text_chunk(client, block, from_lang, to_lang, config)
        
        if translated_piece is not None:
            translated_html_pieces.append(translated_piece)
        else:
            logger.error(f"Failed to translate HTML block {i+1}, using original content for this block.")
            translated_html_pieces.append(block) # Fallback to original block
            
    return "".join(translated_html_pieces)

def translate_epub(
    client: OpenAI, input_epub_path: str, output_epub_path: str, 
    from_lang: str, to_lang: str, config: Dict[str, Any],
    from_chapter_num: int = 1, to_chapter_num: int = float('inf')
) -> None:
    """Translates content of an EPUB book using large HTML block strategy."""
    try:
        book = epub.read_epub(input_epub_path)
    except FileNotFoundError: 
        logger.error(f"Input EPUB file not found: {input_epub_path}")
        sys.exit(1)
    except Exception as e: 
        logger.error(f"Error reading EPUB file '{input_epub_path}': {e}", exc_info=True)
        sys.exit(1)

    items_to_translate = [item for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT)]
    total_chapters_in_book = len(items_to_translate)
    logger.info(f"Found {total_chapters_in_book} document items (chapters/sections) in the EPUB.")

    chapters_to_process_tuples = []
    for idx, item in enumerate(items_to_translate):
        user_facing_num = idx + 1
        if from_chapter_num <= user_facing_num <= to_chapter_num:
            chapters_to_process_tuples.append((idx, item))
    
    actual_chapters_to_process_count = len(chapters_to_process_tuples)
    
    if not chapters_to_process_tuples:
        logger.warning("No chapters fall within the specified --from-chapter and --to-chapter range.")
        try:
            epub.write_epub(output_epub_path, book, {})
            logger.info(f"Output EPUB (no chapters translated) saved to: {output_epub_path}")
        except Exception as e: 
            logger.error(f"Error writing output EPUB to '{output_epub_path}': {e}", exc_info=True)
        return

    logger.info(f"Will attempt to translate {actual_chapters_to_process_count} chapters (from {from_chapter_num} to {min(to_chapter_num, total_chapters_in_book)}).")
    processed_chapter_count = 0

    main_pbar = tqdm(
        iterable=chapters_to_process_tuples,
        total=actual_chapters_to_process_count,
        desc="Translating Chapters", 
        unit="ch", 
        disable=logger.getEffectiveLevel() > logging.INFO,
        smoothing=0.15,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    )

    for original_chapter_idx, item in main_pbar:
        current_chapter_user_num = original_chapter_idx + 1
        main_pbar.set_postfix_str(f"{item.get_name()[:25]}...", refresh=True)
        logger.info(f"Processing chapter {current_chapter_user_num}/{total_chapters_in_book} ('{item.get_name()}')...")
        
        try:
            original_full_html_content = item.get_content().decode('utf-8', errors='replace')
            
            # Try to isolate <body> content for translation, as sending full XHTML (with <head>)
            # might confuse the LLM or be unnecessary.
            soup_parser = BeautifulSoup(original_full_html_content, 'html.parser')
            body_tag = soup_parser.body
            
            html_target_for_translation: str
            is_body_translation = False

            if body_tag:
                # Extract the HTML string of the body's children
                html_target_for_translation = "".join(str(child) for child in body_tag.contents)
                is_body_translation = True
                logger.debug(f"Chapter {current_chapter_user_num}: Extracted <body> content for translation.")
            else:
                # If no <body> tag found (e.g., it's an HTML fragment), translate the whole item content.
                html_target_for_translation = original_full_html_content
                is_body_translation = False
                logger.debug(f"Chapter {current_chapter_user_num}: No <body> tag found, will translate entire item content.")

            logger.debug(f"Chapter {current_chapter_user_num}: HTML content for translation (first 200 chars): {html_target_for_translation[:200]}...")

            translated_html_content_for_target = translate_html_chapter_content(
                html_target_for_translation, client, from_lang, to_lang, config
            )
            
            # Reconstruct the chapter content
            if is_body_translation and body_tag: # Make sure body_tag is not None
                # Parse the translated HTML (which should be the content of the body)
                # and replace the original body's children with these new children.
                # This preserves the original <body> tag itself and its attributes.
                translated_body_children_soup = BeautifulSoup(translated_html_content_for_target, 'html.parser')
                body_tag.clear() # Remove old children from the original body_tag
                for child_node in list(translated_body_children_soup.contents): # list() to copy
                    body_tag.append(child_node) # Append new translated children
                final_content_to_set = str(soup_parser) # Get the string of the modified original soup
            else:
                # If we translated the whole original content (no body was isolated)
                final_content_to_set = translated_html_content_for_target
            
            item.set_content(final_content_to_set.encode('utf-8'))

            logger.debug(f"Chapter {current_chapter_user_num}: Finished processing, content updated.")
            processed_chapter_count +=1

        except Exception as e: 
            logger.error(f"Error processing chapter {current_chapter_user_num} ('{item.get_name()}'): {e}", exc_info=True)
            logger.warning(f"Skipping translation for chapter {current_chapter_user_num} due to error.")
            continue

    if main_pbar:
        main_pbar.close()

    if processed_chapter_count == 0 and actual_chapters_to_process_count > 0 :
        logger.warning("No chapters were successfully processed in this run, though some were in range. Check logs for errors.")
    elif processed_chapter_count > 0:
        logger.info(f"Successfully processed and updated {processed_chapter_count} chapters in this run.")

    try:
        epub.write_epub(output_epub_path, book, {})
        logger.info(f"Translated EPUB successfully saved to: {output_epub_path}")
    except Exception as e: 
        logger.error(f"Error writing translated EPUB to '{output_epub_path}': {e}", exc_info=True)
        sys.exit(1)


# --- show_chapters_info (remains the same as previous versions) ---
def show_chapters_info(input_epub_path: str) -> None:
    try:
        book = epub.read_epub(input_epub_path)
    except FileNotFoundError: 
        logger.error(f"Input EPUB file not found: {input_epub_path}")
        sys.exit(1)
    except Exception as e: 
        logger.error(f"Error reading EPUB file '{input_epub_path}': {e}", exc_info=True)
        sys.exit(1)

    metadata_title_list = book.get_metadata('DC', 'title')
    if metadata_title_list:
        metadata_title = metadata_title_list[0]
        logger.info(f"EPUB Book Title: {metadata_title[0] if isinstance(metadata_title, tuple) else metadata_title}")
    else:
        logger.info("EPUB Book Title: Not found in metadata.")

    metadata_lang_list = book.get_metadata('DC', 'language')
    if metadata_lang_list:
        metadata_lang = metadata_lang_list[0]
        logger.info(f"EPUB Book Language: {metadata_lang[0] if isinstance(metadata_lang, tuple) else metadata_lang}")
    else:
        logger.info("EPUB Book Language: Not found in metadata.")
    
    logger.info("\nTable of Contents (from NCX/Nav metadata if available):")
    if book.toc:
        for toc_item in book.toc:
            if isinstance(toc_item, epub.Link):
                logger.info(f"  - {toc_item.title} (href: {toc_item.href})")
            elif isinstance(toc_item, tuple) and len(toc_item) > 0 and isinstance(toc_item[0], epub.Link): 
                 logger.info(f"  - {toc_item[0].title} (href: {toc_item[0].href}) (has {len(toc_item[1]) if len(toc_item) > 1 and isinstance(toc_item[1], list) else 0} sub-items)")
    else:
        logger.info("  No ToC metadata found or ToC is empty.")

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
            try:
                html_content_for_preview = content.decode('utf-8', errors='replace')
                soup = BeautifulSoup(html_content_for_preview, 'html.parser')
                text_content = soup.get_text(separator=' ', strip=True)
                text_content_cleaned = re.sub(r'\s+', ' ', text_content).strip()
                content_preview = text_content_cleaned[:250] + ('...' if len(text_content_cleaned) > 250 else '')
            except Exception as e_preview: 
                logger.debug(f"Could not fully parse chapter {chapter_num} for preview: {e_preview}")
                content_preview = "[Content preview partially unparseable or error during preview generation]"
        except Exception as e_content: 
            logger.debug(f"Could not get content for chapter {chapter_num} for preview: {e_content}")
            content_preview = "[Content not accessible for preview]"

        logger.info(f"\n▶️  Chapter {chapter_num}/{total_docs}")
        logger.info(f"   Item Name: {item.get_name()}")
        logger.info(f"   File Name: {item.file_name}")
        logger.info(f"   ID: {item.id}")
        logger.info(f"   Media Type: {item.media_type}")
        logger.info(f"   Size (bytes): {char_count}")
        logger.info(f"   Preview: {content_preview}")

# --- Main Execution (remains the same as previous versions) ---
def main():
    parser = argparse.ArgumentParser(description='Translate EPUB books using AI or show chapter information.')
    parser.add_argument('--config', default='config.yaml', help='Path to YAML configuration file (default: config.yaml).')
    parser.add_argument('--debug', action='store_true', help='Force DEBUG logging level for this run, overriding config.')
    
    subparsers = parser.add_subparsers(dest='mode', help='Mode of operation.', required=True)

    parser_translate = subparsers.add_parser('translate', help='Translate an EPUB book.')
    parser_translate.add_argument('--input', required=True, help='Input EPUB file path.')
    parser_translate.add_argument('--output', required=True, help='Output EPUB file path.')
    parser_translate.add_argument('--from-lang', required=True, help='Source language code (e.g., en, es, de).')
    parser_translate.add_argument('--to-lang', required=True, help='Target language code (e.g., en, es, de).')
    parser_translate.add_argument('--from-chapter', type=int, default=1, help='Starting chapter number (1-based) for translation (inclusive).')
    parser_translate.add_argument('--to-chapter', type=int, default=float('inf'), help='Ending chapter number (1-based) for translation (inclusive). Default: all chapters.')

    parser_show = subparsers.add_parser('show-chapters', help='Show chapter information from an EPUB file.')
    parser_show.add_argument('--input', required=True, help='Input EPUB file path.')

    args = parser.parse_args()

    config = read_config(args.config)
    if args.debug:
        config['logging']['level'] = 'DEBUG' 
    setup_logging(config) 
    if args.debug: 
        logger.info("DEBUG logging enabled by command-line argument.")

    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        api_key = config.get("openai", {}).get("api_key")
    
    if args.mode == 'translate':
        configured_model = config.get("openai", {}).get("model", DEFAULT_CONFIG["openai"]["model"])
        if not configured_model: # Check if model is missing or empty string
             logger.critical("CRITICAL CONFIGURATION WARNING: No OpenAI model specified in configuration. Translation will fail.")
             sys.exit(1) # Exit if no model is configured
        # Specific check for the known problematic placeholder if it's still the default
        if configured_model == "gpt-4.1-mini" and DEFAULT_CONFIG["openai"]["model"] == "gpt-4.1-mini": 
            logger.critical(f"CRITICAL CONFIGURATION WARNING: The OpenAI model is effectively '{configured_model}' (from script default). "
                            "This is highly unlikely to be a valid public model name. "
                            "Translation will almost certainly fail. Please set a valid model (e.g., 'gpt-4o-mini') "
                            "in your config.yaml file or by editing DEFAULT_CONFIG in the script.")
        
        logger.info(f"Using OpenAI model: {configured_model}") # Log the model being used

        if not api_key:
            logger.error("OpenAI API key not found. Set OPENAI_API_KEY environment variable or add to config.yaml.")
            sys.exit(1)
        
        try:
            openai_client = OpenAI(api_key=api_key)
            logger.info("OpenAI client initialized.")
        except AuthenticationError as e_auth:
             logger.error(f"OpenAI Authentication Error during client initialization: {e_auth}. Check your API key.", exc_info=True)
             sys.exit(1)
        except Exception as e: 
            logger.error(f"Failed to initialize OpenAI client: {e}", exc_info=True)
            sys.exit(1)

        logger.info(f"Starting EPUB translation from '{args.input}' to '{args.output}'.")
        logger.info(f"Languages: {args.from_lang} -> {args.to_lang}")
        to_chapter_display = 'all' if args.to_chapter == float('inf') else args.to_chapter
        logger.info(f"Chapter range: {args.from_chapter} to {to_chapter_display}")
        
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

if __name__ == "__main__":
    main()
