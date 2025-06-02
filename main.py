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
    SENTENCE_TOKENIZER_AVAILABLE = True
except ImportError:
    nltk = None
    SENTENCE_TOKENIZER_AVAILABLE = False

# --- Configuration ---
DEFAULT_CONFIG = {
    "openai": {
        "api_key": None, 
        "model": "gpt-4o-mini", 
        "temperature": 0.4,
        "max_tokens_per_chunk": 4000,
        "request_timeout": 120, 
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
logger = logging.getLogger(__name__)

# --- Helper Functions ---

def setup_logging(config: Dict[str, Any]) -> None:
    """Sets up global logging and ensures NLTK 'punkt' and 'punkt_tab' are available."""
    log_config = config.get("logging", DEFAULT_CONFIG["logging"])
    level_str = log_config.get("level", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    log_format = log_config.get("format", "%(asctime)s - %(levelname)s - %(message)s")
    logging.basicConfig(level=level, format=log_format, stream=sys.stdout, force=True)
    logger.info(f"Logging configured to level: {level_str}")

    # Silence INFO logs from httpx and httpcore for cleaner output
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    if not SENTENCE_TOKENIZER_AVAILABLE:
        logger.warning("NLTK library not found. Falling back to basic regex-based sentence splitting. "
                       "For better sentence tokenization, please install NLTK: pip install nltk")
    elif nltk:
        resources_to_download = ['punkt']
        # Since 'punkt_tab' worked for you with nltk.download('punkt_tab'),
        # we can add it if it's a recognized standalone download identifier by NLTK.
        # Typically, 'punkt' should cover its components.
        # Let's try 'punkt' first, as it's the main package.
        # If 'punkt_tab' is a separate entity NLTK's downloader recognizes, it can be added.
        # For now, focusing on 'punkt' as the primary resource.
        # If specific 'punkt_tab' issues persist, one might consider adding 'punkt_tab' to resources_to_download.

        for resource_name in resources_to_download:
            try:
                # Check if the primary resource directory exists
                # For 'punkt', this is 'tokenizers/punkt'
                nltk.data.find(f'tokenizers/{resource_name}')
                logger.info(f"NLTK resource '{resource_name}' found.")
            except nltk.downloader.DownloadError: 
                logger.warning(f"NLTK resource '{resource_name}' not found. Attempting download...")
                try:
                    nltk.download(resource_name, quiet=False) 
                    nltk.data.find(f'tokenizers/{resource_name}') # Verify again
                    logger.info(f"NLTK resource '{resource_name}' is now available after download attempt.")
                except Exception as e_nltk_dl:
                    logger.error(f"Failed to download or verify NLTK resource '{resource_name}': {e_nltk_dl}. "
                                 "NLTK sentence tokenization might be impaired.")
            except Exception as e_nltk_find:
                 logger.error(f"An error occurred while checking for NLTK resource '{resource_name}': {e_nltk_find}. "
                              "NLTK sentence tokenization might be impaired.")

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
                     f"This means token counting will be inaccurate and API calls with this model name WILL LIKELY FAIL. "
                     f"Please use a valid OpenAI model name (e.g., 'gpt-4o-mini'). "
                     f"Falling back to 'cl100k_base' for token counting, but this is NOT a fix for the API model name.")
        encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def _nltk_sentence_tokenize(text: str, language: str = 'english') -> List[str]:
    """Tokenizes text into sentences using NLTK. Assumes 'punkt' resources are available."""
    if not nltk or not SENTENCE_TOKENIZER_AVAILABLE:
        logger.error("NLTK sentence tokenizer called directly but NLTK is not available. Falling back to regex.")
        return _regex_sentence_tokenize(text)
    try:
        return nltk.tokenize.sent_tokenize(text, language=language)
    except LookupError as e_lookup: 
        logger.error(f"NLTK LookupError during sentence tokenization for language '{language}': {e_lookup}. "
                     f"This might mean 'punkt' or its language-specific resources are still missing "
                     "despite download attempts in setup_logging. Falling back to regex.")
        return _regex_sentence_tokenize(text)
    except Exception as e:
        logger.error(f"Unexpected error during NLTK sentence tokenization: {e}. Falling back to regex.", exc_info=True)
        return _regex_sentence_tokenize(text)

def _regex_sentence_tokenize(text: str) -> List[str]:
    """Basic regex-based sentence splitter."""
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<=\.|\?|!)\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def get_sentence_tokenizer(language_code: str) -> Callable[[str], List[str]]:
    """Returns the best available sentence tokenizer."""
    if nltk and SENTENCE_TOKENIZER_AVAILABLE:
        lang_map = {
            'en': 'english', 'es': 'spanish', 'fr': 'french', 'de': 'german',
            'it': 'italian', 'pt': 'portuguese', 'nl': 'dutch', 'ru': 'russian'
        }
        nltk_lang = lang_map.get(language_code.lower(), 'english')
        try:
            _nltk_sentence_tokenize("Test sentence.", language=nltk_lang) 
            logger.debug(f"Using NLTK sentence tokenizer for language: {nltk_lang}") # Changed to DEBUG
            return lambda text: _nltk_sentence_tokenize(text, language=nltk_lang)
        except Exception as e_test: 
            logger.warning(f"NLTK tokenizer test for '{nltk_lang}' failed ({e_test}). Defaulting to English NLTK or regex.")
            try:
                _nltk_sentence_tokenize("Test sentence.", language='english')
                logger.debug("Using NLTK sentence tokenizer for language: english (fallback)") # Changed to DEBUG
                return lambda text: _nltk_sentence_tokenize(text, language='english')
            except Exception as e_eng_test:
                logger.error(f"NLTK English tokenizer also failed test ({e_eng_test}). Falling back to regex for all tokenization.")
    
    logger.debug("Using regex-based sentence tokenizer (NLTK not available, not configured for language, or 'punkt' resources missing).") # Changed to DEBUG
    return _regex_sentence_tokenize

def split_plain_text_into_chunks(text: str, max_tokens: int, model_name: str, from_lang_code: str) -> List[str]:
    """Splits plain text into chunks by sentences, keeping each chunk below max_tokens."""
    logger.debug(f"split_plain_text_into_chunks: Input text: '{text[:150]}...'")
    if not text.strip():
        logger.debug("split_plain_text_into_chunks: Input text is empty or whitespace. Returning empty list.")
        return []

    sentence_tokenizer = get_sentence_tokenizer(from_lang_code)
    # Determine tokenizer name for logging (best effort)
    tokenizer_name_for_log = "selected tokenizer"
    if hasattr(sentence_tokenizer, '__name__'):
        tokenizer_name_for_log = sentence_tokenizer.__name__
    elif hasattr(sentence_tokenizer, 'func') and hasattr(sentence_tokenizer.func, '__name__'): # For lambdas
        tokenizer_name_for_log = sentence_tokenizer.func.__name__


    sentences = sentence_tokenizer(text)
    logger.debug(f"split_plain_text_into_chunks: Found {len(sentences)} sentences using {tokenizer_name_for_log}. First three: {sentences[:3]}")
    
    chunks = []
    current_chunk_sentences = []
    current_chunk_tokens = 0

    for i, sentence in enumerate(sentences):
        logger.debug(f"  Processing sentence {i+1}/{len(sentences)}: '{sentence[:70]}...'")
        sentence_tokens = count_tokens(sentence, model_name)
        logger.debug(f"    Sentence tokens: {sentence_tokens}")
        
        if sentence_tokens == 0 and not sentence.strip():
            logger.debug(f"    Skipping empty sentence.")
            continue

        if sentence_tokens > max_tokens:
            logger.warning(f"A single sentence exceeds max_tokens ({sentence_tokens}/{max_tokens}). Splitting it: '{sentence[:50]}...'")
            words = sentence.split()
            temp_sentence_part = ""
            for word_idx, word in enumerate(words):
                next_part = temp_sentence_part + (" " if temp_sentence_part else "") + word
                if count_tokens(next_part, model_name) <= max_tokens:
                    temp_sentence_part = next_part
                else:
                    if temp_sentence_part:
                        logger.debug(f"    Adding oversized sentence part (fit): '{temp_sentence_part[:50]}...' (tokens: {count_tokens(temp_sentence_part, model_name)})")
                        chunks.append(temp_sentence_part)
                    temp_sentence_part = word
                    if count_tokens(temp_sentence_part, model_name) > max_tokens:
                        logger.error(f"A single word is too long for token limit after trying to split sentence: '{word[:50]}...'. Skipping this word.")
                        temp_sentence_part = "" 
            if temp_sentence_part:
                logger.debug(f"    Adding final oversized sentence part: '{temp_sentence_part[:50]}...' (tokens: {count_tokens(temp_sentence_part, model_name)})")
                chunks.append(temp_sentence_part)
            current_chunk_sentences = []
            current_chunk_tokens = 0
            continue

        if current_chunk_tokens + sentence_tokens <= max_tokens:
            current_chunk_sentences.append(sentence)
            current_chunk_tokens += sentence_tokens
            logger.debug(f"    Added sentence to current chunk. New chunk tokens: {current_chunk_tokens}")
        else:
            if current_chunk_sentences:
                chunks.append(" ".join(current_chunk_sentences))
                logger.debug(f"    Finalized chunk ({len(current_chunk_sentences)} sentences). Total chunks: {len(chunks)}")
            current_chunk_sentences = [sentence]
            current_chunk_tokens = sentence_tokens
            logger.debug(f"    Started new chunk with sentence. Chunk tokens: {current_chunk_tokens}")

    if current_chunk_sentences:
        chunks.append(" ".join(current_chunk_sentences))
        logger.debug(f"    Finalized last chunk ({len(current_chunk_sentences)} sentences). Total chunks: {len(chunks)}")
        
    final_chunks = [chunk for chunk in chunks if chunk.strip()]
    logger.debug(f"split_plain_text_into_chunks: Generated {len(final_chunks)} final non-empty chunks.")
    return final_chunks

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
    client: OpenAI, text_chunk: str, from_lang: str, to_lang: str, config: Dict[str, Any]
) -> Optional[str]:
    """Translates a single text chunk using OpenAI API with retries."""
    openai_config = config.get("openai", DEFAULT_CONFIG["openai"])
    translation_config = config.get("translation_settings", DEFAULT_CONFIG["translation_settings"])
    
    model_to_use = openai_config["model"]
    logger.debug(f"translate_single_text_chunk: Attempting to translate chunk using OpenAI model: '{model_to_use}'. Chunk: '{text_chunk[:100]}...'")

    for attempt in range(translation_config["retries"]):
        try:
            logger.debug(f"  API Call Attempt {attempt + 1}/{translation_config['retries']}")
            response = client.chat.completions.create(
                model=model_to_use,
                temperature=openai_config["temperature"],
                messages=[
                    {'role': 'system', 'content': system_prompt(from_lang, to_lang)},
                    {'role': 'user', 'content': text_chunk},
                ],
                timeout=openai_config.get("request_timeout", 60)
            )
            translated_text = response.choices[0].message.content
            logger.debug(f"  API Call Success. Raw translated_text: '{translated_text[:100] if translated_text else 'EMPTY_RESPONSE'}'")
            if translated_text:
                return translated_text.strip()
            else:
                logger.warning(f"Received empty translation from API for chunk: '{text_chunk[:50]}...'")
                return "" 
        except RateLimitError as e:
            logger.warning(f"Rate limit exceeded (attempt {attempt + 1}/{translation_config['retries']}). Retrying in {translation_config['retry_delay_seconds'] * (attempt + 1)}s... Error: {e}")
            time.sleep(translation_config['retry_delay_seconds'] * (attempt + 1))
        except APIError as e:
            if "does not exist" in str(e).lower() or "invalid model" in str(e).lower():
                logger.error(f"FATAL OpenAI API Error: Model '{model_to_use}' likely does not exist or you do not have access. "
                             f"Please check the model name in your configuration. Error: {e}", exc_info=True)
                return None 
            logger.error(f"OpenAI API error (attempt {attempt + 1}/{translation_config['retries']}): {e}. Retrying...", exc_info=True)
            time.sleep(translation_config['retry_delay_seconds'])
        except AuthenticationError as e: 
            logger.error(f"OpenAI Authentication Error: {e}. Please check your API key and organization. Halting translation for this chunk.", exc_info=True)
            return None 
        except Exception as e: 
            logger.error(f"An unexpected error occurred during translation (attempt {attempt + 1}/{translation_config['retries']}): {e}", exc_info=True)
            time.sleep(translation_config['retry_delay_seconds'])
    
    logger.error(f"Failed to translate chunk after {translation_config['retries']} retries: '{text_chunk[:50]}...'")
    return None

def process_and_translate_text_content(
    plain_text: str, client: OpenAI, from_lang: str, to_lang: str, config: Dict[str, Any]
) -> str:
    """Splits plain text, translates chunks, and reassembles them."""
    logger.debug(f"process_and_translate_text_content: Input text: '{plain_text[:100]}...'")
    if not plain_text.strip():
        logger.debug("process_and_translate_text_content: Input text is empty or whitespace only. Returning original.")
        return plain_text

    openai_config = config.get("openai", DEFAULT_CONFIG["openai"])
    chunks = split_plain_text_into_chunks(
        plain_text, openai_config["max_tokens_per_chunk"], openai_config["model"], from_lang
    )
    logger.debug(f"process_and_translate_text_content: Generated {len(chunks)} chunks from plain text.")

    if not chunks:
        logger.debug("process_and_translate_text_content: No chunks generated. Returning original plain_text.")
        return plain_text

    translated_chunks = []
    
    # Only show the chunk progress bar if there's more than one chunk AND we are not in deep debug mode
    show_chunk_pbar = (len(chunks) > 1) and (logger.getEffectiveLevel() > logging.DEBUG)

    chunk_iterator = tqdm(
        chunks, 
        desc="  ↪ Translating sub-chunks", 
        unit="chk", 
        leave=False, 
        disable=not show_chunk_pbar,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    )

    for i, chunk in enumerate(chunk_iterator):
        if not show_chunk_pbar and logger.getEffectiveLevel() <= logging.DEBUG:
             logger.debug(f"    Processing chunk {i+1}/{len(chunks)} for text node: '{plain_text[:30]}...'")

        translated_chunk = translate_single_text_chunk(client, chunk, from_lang, to_lang, config)
        if translated_chunk is not None:
            translated_chunks.append(translated_chunk)
        else:
            logger.error(f"Failed to translate chunk {i+1} from text node '{plain_text[:30]}...', using original: '{chunk[:50]}...'")
            translated_chunks.append(chunk)

    final_translation = " ".join(translated_chunks)
    logger.debug(f"process_and_translate_text_content: Final reassembled translation for text node '{plain_text[:30]}...': '{final_translation[:100]}...'")
    return final_translation

def translate_html_element_content(
    element: BeautifulSoup, client: OpenAI, from_lang: str, to_lang: str, config: Dict[str, Any]
) -> None:
    """Recursively traverses a BeautifulSoup element, translates text nodes, and preserves HTML structure."""
    block_skip_tags = ['script', 'style', 'pre', 'code'] 
    
    element_name_for_log = "UnknownElementType"
    if hasattr(element, 'name') and element.name:
        element_name_for_log = element.name
        if element.name in block_skip_tags:
            logger.debug(f"Skipping translation for content of <{element.name}> tag.")
            return
    elif isinstance(element, NavigableString):
        element_name_for_log = "NavigableStringWrapper"
    
    logger.debug(f"Processing element <{element_name_for_log}> for translation.")
    
    for content_item in list(element.contents): 
        if isinstance(content_item, NavigableString):
            original_text_node_content = str(content_item)
            text_to_translate = original_text_node_content.strip() 
            parent_name_for_log = element_name_for_log 
            logger.debug(f"  Found NavigableString in <{parent_name_for_log}>. Original: '{original_text_node_content[:60]}...', Stripped for check: '{text_to_translate[:60]}...'")

            if text_to_translate: 
                logger.debug(f"    Attempting to translate stripped text: '{text_to_translate[:70]}...'")
                translated_text_segment = process_and_translate_text_content(
                    text_to_translate, client, from_lang, to_lang, config
                )
                logger.debug(f"    Received from translation processing: '{translated_text_segment[:70] if translated_text_segment else 'None/Empty'}'")

                if translated_text_segment is not None and translated_text_segment != text_to_translate:
                    new_node = NavigableString(translated_text_segment) 
                    content_item.replace_with(new_node)
                    logger.debug(f"    Replaced content in <{parent_name_for_log}> with translated text.")
                elif translated_text_segment == text_to_translate:
                    logger.debug(f"    Translation returned identical text. No replacement for: '{text_to_translate[:50]}...'")
                else: 
                    logger.warning(f"    Translation failed or returned empty for: '{text_to_translate[:50]}...'. Original content kept, or replaced with empty if API returned empty.")
                    if translated_text_segment == "": 
                        content_item.replace_with(NavigableString("")) 
            else:
                logger.debug(f"    Skipping translation for fully whitespace NavigableString: '{original_text_node_content[:60]}'")
        
        elif hasattr(content_item, 'name') and content_item.name: 
            logger.debug(f"  Recursively calling translate_html_element_content for child <{content_item.name}>")
            translate_html_element_content(content_item, client, from_lang, to_lang, config)
        else:
            logger.debug(f"  Skipping non-Tag, non-NavigableString content_item: {type(content_item)} ('{str(content_item)[:60]}...')")

def translate_epub(
    client: OpenAI, input_epub_path: str, output_epub_path: str, 
    from_lang: str, to_lang: str, config: Dict[str, Any],
    from_chapter_num: int = 1, to_chapter_num: int = float('inf')
) -> None:
    """Translates content of an EPUB book and writes the output to a new file."""
    try:
        book = epub.read_epub(input_epub_path)
    except FileNotFoundError: 
        logger.error(f"Input EPUB file not found: {input_epub_path}")
        sys.exit(1)
    except Exception as e: 
        logger.error(f"Error reading EPUB file '{input_epub_path}': {e}", exc_info=True)
        sys.exit(1)

    items_to_translate = [item for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT)]
    total_chapters_in_book = len(items_to_translate) # Total in book
    logger.info(f"Found {total_chapters_in_book} document items (chapters/sections) in the EPUB.")

    # Filter items based on user-specified chapter range for processing
    chapters_to_process_tuples = [] # Store (original_index, item)
    for idx, item in enumerate(items_to_translate):
        user_facing_num = idx + 1
        if from_chapter_num <= user_facing_num <= to_chapter_num:
            chapters_to_process_tuples.append((idx, item))
    
    actual_chapters_to_process_count = len(chapters_to_process_tuples)
    
    if not chapters_to_process_tuples:
        logger.warning("No chapters fall within the specified --from-chapter and --to-chapter range.")
        # Still try to write the book, which will be an identical copy if no chapters processed
        try:
            epub.write_epub(output_epub_path, book, {})
            logger.info(f"Output EPUB (no chapters translated) saved to: {output_epub_path}")
        except Exception as e: 
            logger.error(f"Error writing output EPUB to '{output_epub_path}': {e}", exc_info=True)
        return

    logger.info(f"Will attempt to translate {actual_chapters_to_process_count} chapters (from {from_chapter_num} to {min(to_chapter_num, total_chapters_in_book)}).")

    processed_chapter_count = 0 # Counts successfully processed chapters in the current run

    # --- TQDM MAIN CHAPTER PROGRESS BAR SETUP ---
    main_pbar = tqdm(
        iterable=chapters_to_process_tuples, # Iterate over the pre-filtered list of (index, item)
        total=actual_chapters_to_process_count, # Set total to the number of chapters we will process
        desc="Translating Chapters", 
        unit="chapter", 
        disable=logger.getEffectiveLevel() > logging.INFO, # Disable if not INFO level or higher
        smoothing=0.15, # More smoothing for ETA
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'
    )
    # --- END OF TQDM SETUP ---

    for original_chapter_idx, item in main_pbar:
        current_chapter_user_num = original_chapter_idx + 1 # User-facing number based on original index
        
        main_pbar.set_postfix_str(f"{item.get_name()[:25]}...", refresh=True) # Update for immediate display

        # Log which chapter is being processed (user-facing number)
        logger.info(f"Processing chapter {current_chapter_user_num}/{total_chapters_in_book} ('{item.get_name()}')...")
        
        try:
            original_html_content = item.get_content().decode('utf-8', errors='replace')
            soup = BeautifulSoup(original_html_content, 'html.parser')
            
            target_element_to_translate = soup.body if soup.body else soup
            if target_element_to_translate:
                target_element_name_for_log = target_element_to_translate.name if hasattr(target_element_to_translate, 'name') else 'document_root'
                logger.debug(f"Chapter {current_chapter_user_num}: Translating content within <{target_element_name_for_log}>.")
                translate_html_element_content(target_element_to_translate, client, from_lang, to_lang, config)
                item.set_content(str(soup).encode('utf-8'))
                logger.debug(f"Chapter {current_chapter_user_num}: Finished processing, content updated.")
            else: 
                logger.warning(f"Chapter {current_chapter_user_num} ('{item.get_name()}') has no 'body' tag or parsable root content to translate.")
            processed_chapter_count +=1

        except Exception as e: 
            logger.error(f"Error processing chapter {current_chapter_user_num} ('{item.get_name()}'): {e}", exc_info=True)
            logger.warning(f"Skipping translation for chapter {current_chapter_user_num} due to error.")
            continue # Skip to next item in main_pbar

    if main_pbar: # Ensure bar is closed
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

def show_chapters_info(input_epub_path: str) -> None:
    """Displays information about each chapter/document item in an EPUB book."""
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

# --- Main Execution ---
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
        if configured_model == "gpt-4.1-mini" and DEFAULT_CONFIG["openai"]["model"] == "gpt-4.1-mini": 
            logger.critical(f"CRITICAL CONFIGURATION WARNING: The OpenAI model is effectively '{configured_model}' (from script default). "
                            "This is highly unlikely to be a valid public model name. "
                            "Translation will almost certainly fail. Please set a valid model (e.g., 'gpt-4o-mini') "
                            "in your config.yaml file or by editing DEFAULT_CONFIG in the script.")
        elif not configured_model:
             logger.critical("CRITICAL CONFIGURATION WARNING: No OpenAI model specified in configuration. Translation will fail.")

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
