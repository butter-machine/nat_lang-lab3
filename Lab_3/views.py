import os
from operator import itemgetter

from django.conf import settings
from django.core.files.base import ContentFile
from django.views.generic import TemplateView

from Lab_3.services.language import StatisticsService
from Lab_3.settings import FILES_DIR


class IndexView(TemplateView):
    template_name = 'index.html'

    def get_context_data(self, **kwargs):
        return {'files': os.listdir(FILES_DIR)}


class AbstractView(TemplateView):
    template_name = 'abstract.html'

    def get_context_data(self, **kwargs):
        file_path = self.kwargs['document']
        with open(os.path.join(FILES_DIR, file_path), 'r') as target_file:
            content = target_file.read()
            statistics_service = StatisticsService(content)

            statistics_service.language_service.detect_lang(content)
            content_file = ContentFile(content, name=target_file.name)
            target_file.seek(0)
            sentences = []
            for index, sentence in enumerate(statistics_service.language_service.sentences(content)):
                sentence_text = sentence
                sentence = statistics_service.language_service.tokenize(sentence)
                weight = statistics_service.tfidf(sentence, content_file)\
                    * statistics_service.posd(content, sentence_text)\
                    * statistics_service.posp(content, sentence_text)
                sentences.append({'sentence': sentence_text, 'weight': weight, 'index': index})

            abstract = sorted(sentences, key=lambda x: x['weight'])[:settings.SENTENCE_COUNT]
            abstract = sorted(abstract, key=lambda x: x['index'])
            abstract = '\n'.join(list(map(itemgetter('sentence'), abstract)))
        return {'abstract': abstract}
