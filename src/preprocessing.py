import re
import string
import itertools
import random
import math

from collections import namedtuple
from functools import partial
from typing import List
from pyparsing import nums, nestedExpr, oneOf, delimitedList, printables, Combine, Word
from pyparsing import Optional

import spacy
import numpy as np

from glove import Glove, Corpus
from sklearn.feature_extraction.text import TfidfVectorizer
from wmd import WMD

from utils import merge_lists_alternating


class Preprocessor:
    """
    Class Preprocessor implements all necessary operations to prepare raw
    text input for modeling
    """
    WordLemma = namedtuple(
        'WordLemma', ['start_char', 'end_char', 'text', 'label_']
    )  # proxy for a class representing a text span with a label

    WordEmbedding = namedtuple('WordEmbedding', ['idf_id', 'glove_id'])  # necessary for determining
    # correct weights for embedding vectors

    def __init__(
            self, glove_components=300, min_df=5, max_df=0.4):
        self.glove_model = Glove(no_components=glove_components)
        self.tf_idf_model = TfidfVectorizer(
            min_df=min_df, max_df=max_df,
            token_pattern='[^\s]+', lowercase=False
        )
        self.word_mapping = None
        self.embedding_dim = glove_components
        self.wmd = None
        self._r = None

    def preprocess(self, text: str) -> str:
        raise NotImplementedError

    def sentence_tokenizer(self, text: str) -> List[str]:
        raise NotImplementedError

    def fit_glove(self, sentences, window, epochs):
        corpus = Corpus()
        corpus.fit(sentences, window=window)
        self.glove_model.fit(corpus.matrix, epochs=epochs, no_threads=8)
        self.glove_model.add_dictionary(corpus.dictionary)

    def fit_tf_idf(self, articles):
        self.tf_idf_model.fit(articles)

    def fit(
            self, inputs, return_clean=True,
            clean=True, window=10, epochs=100):
        if clean:
            print('Cleaning {n_inputs} inputs...'.format(
                n_inputs=len(inputs)), end='')
            clean_inputs = [self.preprocess(input) for input in inputs]
            print('Done!')
        else:
            clean_inputs = inputs[:]
        print('Training Tf-idf model...', end='')
        self.fit_tf_idf(clean_inputs)
        print('Done!')
        sentences_per_input = [self.sentence_tokenizer(input) for input in clean_inputs]
        sentences = itertools.chain.from_iterable(sentences_per_input)
        tokenized_sentences = [sentence.split() for sentence in sentences]
        print('Training Glove model...', end='')
        self.fit_glove(tokenized_sentences, window=window, epochs=epochs)
        print('Done!')
        valid_words = set.intersection(
            set(self.glove_model.dictionary.keys()),
            set(self.tf_idf_model.vocabulary_.keys()))
        self.word_mapping = {
            word: self.WordEmbedding(
                glove_id=self.glove_model.dictionary[word],
                idf_id=self.tf_idf_model.vocabulary_[word]
            ) for word in valid_words
        }
        if return_clean:
            return clean_inputs

    def article_to_input(self, article):
        tokens = article.split()
        word_embeddings = [
            self.word_mapping[token] for token in tokens
            if token in self.word_mapping
        ]
        weight_ids = [(we.glove_id, we.idf_id) for we in word_embeddings]
        glove_ids, idf_ids = zip(*weight_ids)
        words = np.array(glove_ids, dtype=np.uint32)
        weights = np.array([
            self.tf_idf_model.idf_[idf_id]
            for idf_id in idf_ids
        ], dtype=np.float32)
        return words, weights

    def _single_embed(self, article, embedding_function, preprocess):
        if preprocess:
            article = self.clean(article)
        try:
            words, weights = self.article_to_input(article)
        except ValueError:
            print('Empty embedding\n\n', article)
            return np.zeros(shape=(self.embedding_dim,))
        return embedding_function(words, weights)

    def _embed(self, inputs, embedding_function, preprocess):
        if isinstance(inputs, list):
            return np.array([
                self._single_embed(input, embedding_function, preprocess)
                for input in inputs
            ])
        return self._single_embed(inputs, embedding_function, preprocess)

    def _idf_embedding(self, words, weights):
        word_vectors = np.array([
            self.glove_model.word_vectors[glove_id]
            for glove_id in words
        ])
        idf_weights = weights / np.sum(weights)
        return np.dot(idf_weights, word_vectors)

    def idf_embed(self, article, preprocess=False):
        return self._embed(
            article, embedding_function=self._idf_embedding,
            preprocess=preprocess)

    def fit_wme_model(self, d_max=6, r=1024):
        self._r = r
        possible_words = list(self.word_mapping)
        nbow = {}
        for i in range(r):
            d = random.sample(range(1, d_max + 1), 1)[0]
            random_doc = random.sample(possible_words, d)
            doc_embeddings = [self.word_mapping[word] for word in random_doc]
            document, idf_ids = zip(*[
                (word.glove_id, word.idf_id) for word in doc_embeddings
            ])
            words = np.array(document, dtype=np.uint32)
            idf_weights = np.array([
                self.tf_idf_model.idf_[idf_id]
                for idf_id in idf_ids
            ], dtype=np.float32)
            weights = idf_weights
            doc_id = '#' + str(i + 1)
            nbow[doc_id] = (doc_id, words, weights)
        self.wmd = WMD(
            embeddings=self.glove_model.word_vectors.astype(np.float32),
            nbow=nbow, vocabulary_min=1
        )

    def _wme_embedding(self, words, weights, gamma):
        distances = np.array([
            self.wmd._WMD_batch(words, weights, '#' + str(i + 1))
            for i in range(self._r)
        ])
        return 1 / math.sqrt(self._r) * np.exp(-gamma * distances)

    def wme_embed(self, article, preprocess=False, gamma=0.19):
        embedding_function = partial(self._wme_embedding, gamma=gamma)
        return self._embed(
            article, embedding_function=embedding_function,
            preprocess=preprocess)


class TextPreprocessor(Preprocessor):
    """
    Class responsible for preprocessing raw text input
    """
    WordLemma = namedtuple(
        'WordLemma', ['start_char', 'end_char', 'text', 'label_']
    )

    def __init__(
            self, glove_components: int=300, min_df: int=5, max_df: float=0.4, min_word_len: int=3):
        super().__init__(glove_components=glove_components, min_df=min_df, max_df=max_df)
        self._nlp = spacy.load('en_core_web_lg')
        self.min_word_len = min_word_len

    def sentence_tokenizer(self, text: str) -> List[str]:
        return text.split('.')

    @classmethod
    def _get_word_lemmas(cls, tokenized_text: spacy.tokens.Doc) -> List[WordLemma]:
        """
        Locates all tokens that need to be replaced with their lemmatized form
        :param tokenized_text: result of a call of nlp instance
        :return: word_lemmas: List; each element is a WordLemma instance, which
            carries information of the token's location in text and what it
            needs to be replaced with
        """
        word_lemmas = []
        doc_length = len(tokenized_text)
        for token in tokenized_text:
            if token.lemma_ != token.text and not token.ent_type:
                trailing = ' ' if token.idx < doc_length - 1\
                                  and '\'' in token.nbor().text else ''
                leading = ' ' if '\'' in token.text else ''
                word_lemma = cls.WordLemma(
                    start_char=token.idx,
                    end_char=token.idx + len(token.text),
                    text=token.text,
                    label_=leading + token.lemma_ + trailing,
                )
                word_lemmas.append(word_lemma)
        return word_lemmas

    @classmethod
    def _split_text_by_ents(cls, text: str, entities: List[WordLemma]) -> List[str]:
        """
        Splits text into chunks that are separated by entities that will be later replaced by
        special tokens
        :param entities: entities that carry information of span in text and the special token label
            to replace the text with
        :return: list of chunks with entities removed
        """
        first_entity_start = entities[0].start_char
        text_parts = [text[:first_entity_start]]
        for i, entity in enumerate(entities[:-1]):
            start_index = entity.end_char
            stop_index = entities[i + 1].start_char
            text_part = text[start_index:stop_index]
            text_parts.append(text_part)
        last_entity_stop = entities[-1].end_char
        text_parts.append(text[last_entity_stop:])
        return text_parts

    @classmethod
    def _tokenize_entities(cls, text: str, entities: List[WordLemma]) -> str:
        """
        Replaces entities with special tokens, e.g. Yesterday -> DATE
        :param text: raw string
        :param entities: entities that carry information of span in text and the special token label
            to replace the text with
        :return: string with replaced entities
        """
        if not entities:  # if the list of entities is empty, do nothing
            return text
        text_parts = cls._split_text_by_ents(text, entities)
        entities_labels = [entity.label_ for entity in entities]
        result_text = merge_lists_alternating(text_parts, entities_labels)
        return ''.join(result_text)

    def preprocess(self, text: str) -> str:
        """
        Performs all necessary operations to transform raw text into a space-separated string of
            ready to use tokens
        :param text: raw string
        :return: cleaned string
        """
        cleaned_text = re.sub(
            '<.*?>', '', text)  # remove html tags
        cleaned_text = re.sub(
            '\n', ' ', cleaned_text)  # remove new line character
        cleaned_text = re.sub(
            '\d', '', cleaned_text)  # remove digits
        punctuation = re.sub(
            '\.|-', '', string.punctuation)
        cleaned_text = re.sub(
            '[' + punctuation + ']', '', cleaned_text)  # remove punctuation
        cleaned_text = re.sub(
            r'\s+', ' ', cleaned_text)  # remove unnecessary whitespaces
        tokenized_text = self._nlp(cleaned_text)
        entities = [
            entity for entity in tokenized_text.ents
            if entity.label_ in {
                'DATE', 'CARDINAL', 'ORDINAL', 'GPE', 'NORP', 'PERSON'
            }
        ]
        word_lemmas = self._get_word_lemmas(tokenized_text)
        full_entities = list(entities) + word_lemmas
        sorted_entities = sorted(full_entities, key=lambda x: x.start_char)
        text_tokenized_entities = self._tokenize_entities(
            cleaned_text, sorted_entities)
        words = text_tokenized_entities.split()
        cleaned_text = ' '.join([word for word in words if len(word) >= self.min_word_len])
        return cleaned_text


class SMILESPreprocessor(Preprocessor):
    """
    Class responsible for preprocessing SMILES strings
    """
    elements = ['c', 'n', 'o', 's', 'p', 'B', 'C', 'N', 'O', 'F', 'P', 'S', 'I', 'Cl', 'Br']
    single_element = oneOf(elements)
    single_compund = nestedExpr('[', ']')
    quantifier = Word(nums)
    non_compound = oneOf(list(printables))  # any character not specified above
    single_token = Combine((single_element | single_compund) + Optional(quantifier))
    full_string = delimitedList(single_token | non_compound)

    def sentence_tokenizer(self, text: str) -> List[str]:
        return [text]

    def preprocess(self, text: str) -> str:
        """
        Transforms a SMILES string into a space separated string of tokens
        """
        tokens = self.full_string.scanString(text)
        tokens = [el[0][0] for el in tokens]
        return ' '.join(tokens)
