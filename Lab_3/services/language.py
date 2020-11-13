import os
from math import log
from typing import List, Any

import nltk
from django.core.files.base import ContentFile
from nltk.corpus import stopwords

from Lab_3.settings import FILES_DIR


class LanguageService:
    def __init__(self):
        self.sentence_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        self.tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
        self.stemmer = nltk.PorterStemmer()

    @staticmethod
    def paragraphs(text: str):
        return text.split('\n')

    def sentences(self, text: str):
        return self.sentence_tokenizer.tokenize(text)

    @staticmethod
    def _clean_tokens(tokens: List[str]):
        stop_words = set(stopwords.words('english'))
        return [token.lower() for token in tokens if token not in stop_words]

    def _stem_tokens(self, tokens: list):
        return [self.stemmer.stem(token) for token in tokens]

    def tokenize(self, text: str):
        tokens = self.tokenizer.tokenize(text)
        tokens = self._clean_tokens(tokens)
        tokens = self._stem_tokens(tokens)

        return [token.lower() for token in tokens]


class StatisticsService:
    def __init__(self):
        self.language_service = LanguageService()

    @staticmethod
    def _sentence_position_function(relative_text: str, sentence: str):
        return 1 - (len(relative_text.partition(sentence)) / len(relative_text))

    @staticmethod
    def _frequency(measurable_value: Any, relative: List[Any]):
        return relative.count(measurable_value) / len(relative)

    @staticmethod
    def _open_file(file_path):
        with open(file_path, 'r') as f:
            content_file = ContentFile(f.read(), name=f.name)
            f.seek(0)

        return content_file

    def posd(self, document_text: str, sentence: str):
        return self._sentence_position_function(document_text, sentence)

    def posp(self, paragraph_text: str, sentence: str):
        return self._sentence_position_function(paragraph_text, sentence)

    def tfidf(self, sentence: List[str], document: ContentFile):
        documents_terms = dict()
        list_dir = os.listdir(FILES_DIR)
        for filename in list_dir:
            content_file = self._open_file(os.path.join(FILES_DIR, filename))
            document_content = content_file.read()
            documents_terms[document] = self.language_service.tokenize(document_content)

        current_document_terms = self.language_service.tokenize(document.read())
        document.seek(0)
        max_term_frequency = max([self._frequency(term, current_document_terms) for term in current_document_terms])
        documents_count = len(list_dir)
        score = 0
        for term in sentence:
            current_document_term_frequency = self._frequency(term, current_document_terms)
            documents_with_current_term = len([key for key in documents_terms.keys() if term in set(documents_terms[document])])
            try:
                score += self._frequency(term, sentence) * 0.5 * (1 + current_document_term_frequency / max_term_frequency)\
                    * log(documents_count / documents_with_current_term)
            except ZeroDivisionError:
                # Sentence tokenizer don't clean and stem tokens so tokens from tokenizer can not be in sentence
                # tokenizer result
                pass

        return score
