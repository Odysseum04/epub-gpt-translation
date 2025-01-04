# 🌎 This repository is a fork of jb41/translate-book

# Translate books with GPT

This project harnesses the power of GPT-4o-mini LLM to translate eBooks from any language into your preferred language, maintaining the integrity and structure of the original content. Imagine having access to a vast world of literature, regardless of the original language, right at your fingertips.

This tool not only translates the text but also carefully compiles each element of the eBook – chapters, footnotes, and all – into a perfectly formatted EPUB file. We use the `gpt-4o-mini` model by default to ensure high-quality translations. However, we understand the need for flexibility, so we've made it easy to switch models in `main.py` according to your specific needs.

## Cost

The cost of translations is pretty low; here is the pricing as 04 january 2025 for gpt-4o-mini:

```bash
$0.225 per 1M input tokens
$0.600 per 1M output tokens
```


The average web novel chapter contain about 2500 tokens (~9000 characters according to https://tokencounter.org/fr). 
### Wich means that translating 400 chapters would cost you about 0.825€, a bit less than 1€.

## 🛠️ Installation

To install the necessary components for our project, follow these simple steps:

```bash
git clone https://github.com/Odysseum04/novel-gpt-translation
```

### Remember to add your OpenAI key to `config.yaml.example`.

```bash
pip install -r requirements.txt
cp config.yaml.example config.yaml
```


## 🎮 Usage

The script comes with a variety of parameters to suit your needs. Here's how you can make the most out of it:

### Show Chapters

Before diving into translation, it's recommended to use the `show-chapters` mode to review the structure of your book:

```bash
python main.py show-chapters --input yourbook.epub
```

This command will display all the chapters, helping you to plan your translation process effectively.

### Translate Mode

#### Basic Usage

To translate a book from English to French, use the following command:

```bash
python main.py translate --input yourbook.epub --output translatedbook.epub --config config.yaml --from-chapter 13 --to-chapter 37 --from-lang EN --to-lang FR
```




## 🤝 Source
This repository is a fork of jb41/translate book, don't forget to thank him by starring his github repo too !
