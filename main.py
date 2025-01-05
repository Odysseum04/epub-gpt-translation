import argparse
import re
import yaml

import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from openai import OpenAI  # Import the OpenAI module

def read_config(config_file: str) -> dict:
    """
    Reads a YAML configuration file and returns its contents as a dictionary.

    Parameters:
    config_file (str): The path to the YAML configuration file.

    Returns:
    dict: Configuration settings as a dictionary.
    """
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    return config

def split_html_by_sentence(html_str: str, max_chunk_size: int = 10000) -> list:
    """
    Splits an HTML string into chunks by sentences, keeping each chunk below a specified size.

    Parameters:
    html_str (str): The HTML string to split.
    max_chunk_size (int): Maximum size for each chunk.

    Returns:
    list: A list of HTML string chunks.
    """
    sentences = html_str.split('. ')
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) > max_chunk_size:
            chunks.append(current_chunk)
            current_chunk = sentence
        else:
            if current_chunk:  # Avoid adding initial dot
                current_chunk += '. '
            current_chunk += sentence

    if current_chunk:
        chunks.append(current_chunk)

    # Ensure every chunk has a trailing dot
    chunks = [chunk.strip() + '.' for chunk in chunks if chunk]

    return chunks

def system_prompt(from_lang: str, to_lang: str) -> str:
    """
    Generates a system prompt for the translation task.

    Parameters:
    from_lang (str): Source language code.
    to_lang (str): Target language code.

    Returns:
    str: System prompt for translation.
    """
    return (
        f"You are an {from_lang}-to-{to_lang} specialized translator. "
        f"Keep all special characters and HTML tags as in the source text. "
        f"Your translation should be in {to_lang} only. "
        f"Ensure the translation is comfortable to read by avoiding overly literal translations. "
        f"Maintain readability and consistency with the source text."
    )

def translate_chunk(client, text: str, from_lang: str, to_lang: str) -> str:
    """
    Translates a given text chunk using an AI language model.

    Parameters:
    client: OpenAI client instance with appropriate API key.
    text (str): Text chunk to translate.
    from_lang (str): Source language code.
    to_lang (str): Target language code.

    Returns:
    str: Translated text chunk.
    """
    response = client.chat.completions.create(
        model='gpt-4o-mini',
        temperature=0.3,
        messages=[
            {'role': 'system', 'content': system_prompt(from_lang, to_lang)},
            {'role': 'user', 'content': text},
        ]
    )

    translated_text = response.choices[0].message.content
    return translated_text

def translate_text(client, text: str, from_lang: str, to_lang: str) -> str:
    """
    Translates a complete text by splitting it into manageable chunks.

    Parameters:
    client: OpenAI client for connecting to translation service.
    text (str): Text to be translated.
    from_lang (str): Source language code.
    to_lang (str): Target language code.

    Returns:
    str: Fully translated text.
    """
    translated_chunks = []
    chunks = split_html_by_sentence(text)

    for i, chunk in enumerate(chunks):
        print(f"\tTranslating chunk {i+1}/{len(chunks)}...")
        translated_chunks.append(translate_chunk(client, chunk, from_lang, to_lang))

    return ' '.join(translated_chunks)

def translate(client, input_epub_path: str, output_epub_path: str, from_lang: str, to_lang: str,
              from_chapter: int = 0, to_chapter: int = 9999) -> None:
    """
    Translates content of an EPUB book and writes the output to a new file.

    Parameters:
    client: OpenAI client instance.
    input_epub_path (str): Path to the input EPUB file.
    output_epub_path (str): Path to the output EPUB file.
    from_lang (str): Source language code.
    to_lang (str): Target language code.
    from_chapter (int): Starting chapter for translation.
    to_chapter (int): Ending chapter for translation.
    """
    book = epub.read_epub(input_epub_path)
    current_chapter = 1
    chapters_count = sum(1 for item in book.get_items() if item.get_type() == ebooklib.ITEM_DOCUMENT)

    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            if from_chapter <= current_chapter <= to_chapter:
                print(f"Processing chapter {current_chapter}/{chapters_count}...")
                soup = BeautifulSoup(item.content, 'html.parser')
                translated_text = translate_text(client, str(soup), from_lang, to_lang)
                item.content = translated_text.encode('utf-8')
            current_chapter += 1

    epub.write_epub(output_epub_path, book, {})

def show_chapters(input_epub_path: str) -> None:
    """
    Displays the beginning of each chapter in an EPUB book.

    Parameters:
    input_epub_path (str): Path to the input EPUB file.
    """
    book = epub.read_epub(input_epub_path)
    current_chapter = 1
    chapters_count = sum(1 for item in book.get_items() if item.get_type() == ebooklib.ITEM_DOCUMENT)

    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            print(f"▶️  Chapter {current_chapter}/{chapters_count} ({len(item.content)} characters)")
            soup = BeautifulSoup(item.content, 'html.parser')
            chapter_beginning = soup.text[:250]
            chapter_beginning = re.sub(r'\n{2,}', '\n', chapter_beginning)
            print(chapter_beginning + "\n\n")
            current_chapter += 1

def main():
    """Main function to parse arguments and execute the appropriate command."""
    parser = argparse.ArgumentParser(description='App to translate or show chapters of a book.')
    subparsers = parser.add_subparsers(dest='mode', help='Mode of operation.')

    parser_translate = subparsers.add_parser('translate', help='Translate a book.')
    parser_translate.add_argument('--input', required=True, help='Input EPUB file path.')
    parser_translate.add_argument('--output', required=True, help='Output EPUB file path.')
    parser_translate.add_argument('--config', required=True, help='Configuration file path.')
    
    parser_translate.add_argument('--from-chapter', type=int, default=1, help='Starting chapter for translation.')
    parser_translate.add_argument('--to-chapter', type=int, default=999999, help='Ending chapter for translation.')
    parser_translate.add_argument('--from-lang', required=True, help='Source language.')
    parser_translate.add_argument('--to-lang', required=True, help='Target language.')

    parser_show = subparsers.add_parser('show-chapters', help='Show the list of chapters.')
    parser_show.add_argument('--input', required=True, help='Input EPUB file path.')

    args = parser.parse_args()

    if args.mode == 'translate':
        config = read_config(args.config)
        openai_client = OpenAI(api_key=config['openai']['api_key'])
        translate(openai_client, args.input, args.output, args.from_lang, args.to_lang, args.from_chapter, args.to_chapter)

    elif args.mode == 'show-chapters':
        show_chapters(args.input)

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
