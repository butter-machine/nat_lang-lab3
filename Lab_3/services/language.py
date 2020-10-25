import nltk


class LanguageService:
    def __init__(self):
        self.sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    @staticmethod
    def paragraphs(text: str):
        return text.split('\n')

    def sentences(self, text: str):
        return self.sentence_tokenizer.tokenize(text)
