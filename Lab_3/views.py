import os
from operator import itemgetter

from django.conf import settings
from django.core.files.base import ContentFile
from django.views.generic import TemplateView

from Lab_3.services.language import LanguageService, StatisticsService
from Lab_3.settings import FILES_DIR


class IndexView(TemplateView):
    template_name = 'index.html'

    def get_context_data(self, **kwargs):
        return {'files': os.listdir(FILES_DIR)}


class AbstractView(TemplateView):
    template_name = 'abstract.html'
    language_service = LanguageService()
    statistics_service = StatisticsService()

    def get_context_data(self, **kwargs):
        file_path = self.kwargs['document']
        with open(os.path.join(FILES_DIR, file_path), 'r') as target_file:
            content = target_file.read()
            content_file = ContentFile(content, name=target_file.name)
            target_file.seek(0)
            sentences = []
            for sentence in self.language_service.sentences(content):
                sentence_text = sentence
                sentence = self.language_service.tokenize(sentence)
                weight = self.statistics_service.tfidf(sentence, content_file)\
                    * self.statistics_service.posd(content, sentence_text)\
                    * self.statistics_service.posp(content, sentence_text)
                sentences.append({'sentence': sentence_text, 'weight': weight})

            abstract = sorted(sentences, key=lambda x: x['weight'])[:settings.SENTENCE_COUNT]
            abstract = '\n'.join(list(map(itemgetter('sentence'), abstract)))
        return {'abstract': abstract}
