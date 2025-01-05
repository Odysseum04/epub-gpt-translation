# 🌎 This repository is a fork of jb41/translate-book

# Translate books with GPT

This project harnesses the power of GPT-4o-mini LLM to translate eBooks from any language into your preferred language, maintaining the integrity and structure of the original content. Imagine having access to a vast world of literature, regardless of the original language, right at your fingertips.

This tool not only translates the text but also carefully compiles each element of the eBook – chapters, footnotes, and all – into a perfectly formatted EPUB file. We use the `gpt-4o-mini` model by default to ensure high-quality translations. However, we understand the need for flexibility, so we've made it easy to switch models in `main.py` according to your specific needs.


## 💵 Cost

The cost of translations is pretty low; here is the pricing as 04 january 2025 for gpt-4o-mini:

```bash
$0.225 per 1M input tokens
$0.600 per 1M output tokens
```


The average web novel chapter contain about 2500 tokens (~9000 characters according to https://tokencounter.org/fr). 
### Wich means that translating 400 chapters would cost you about 0.825€, a bit less than 1€.


## 🛠️ Installation on Windows

Please create a new folder on you machine (I recommend going to your documents forlder and creating a new folder called "novel-translation")
Now go in the search bar of your file browser and copy the way to the folder;
mine looked like this: C:\Users\cleme\OneDrive\Documents\novel-translation\
![image](https://github.com/user-attachments/assets/c8a055d3-40c8-40eb-bf75-142e3ef80924)


Now go to your command invite (cmd) and use the following command:

```bash
cd "THE TEXT YOU JUST COPIED"
```


Example: ![image](https://github.com/user-attachments/assets/df074010-a5b2-4d40-9aca-6661654a6919)


### DO NOT forget to use the "".


Now use this command in your terminal:

```bash
git clone https://github.com/Odysseum04/novel-gpt-translation
```


There is now a new folder inside novel-translation called novel-gpt-translation.


![image](https://github.com/user-attachments/assets/c376eb2d-74ed-4c53-a2af-7b10af2b5dbb)


Open this forlder inside Visual Studio Code:


![image](https://github.com/user-attachments/assets/6ed0c2a4-75b9-4eaf-8849-1f6831c10a7f)




### Remember to add your OpenAI key to `config.yaml.example`.
![image](https://github.com/user-attachments/assets/9d2120b2-56bd-48b0-9e01-5847b9781960)


Tutorial to get an api key:
https://www.youtube.com/watch?v=nafDyRsVnXU&t=18s&ab_channel=TutorialsHubbyFuelYourDigital


When you finished putting the Openai api key inside config.yaml.example, create a new terminal inside visual studio code.


### Use ctrl + shift + p and in the upper area of visual studio code and type "create environment" to create a new venv environment.
![image](https://github.com/user-attachments/assets/a119c64b-3313-4bce-a644-8932e49f637a)
![image](https://github.com/user-attachments/assets/6ac17cf3-8b21-4c80-b61c-8749b0ba6b79)


When you finished putting the Openai api key inside config.yaml.example, create a new terminal inside visual studio code.
![image](https://github.com/user-attachments/assets/a09763a7-115b-4782-b24b-c630d5e8d3f4)


Now in the terminal:
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
