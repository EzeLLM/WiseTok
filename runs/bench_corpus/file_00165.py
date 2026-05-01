from typing import (
    List,
    Tuple,
    Dict,
    Optional,
)

import os
import sys

from dpd.constants import STOP_WORDS

from .types import (
    AnnotatedDataType,
)

class BIOConverter(object):
    def __init__(
        self,
        binary_class: str,
    ):
        self.binary_class = binary_class
    
    def stop_word_heuristic(self, sentence: List[str], predictions: List[str], class_tag: str) -> List[str]:
        proc_pred = list(predictions) # create copy
        start_stop_words = None
        contains_stop_word = False
        stop_word_ranges = []
        for i, (w, p_i) in enumerate(zip(sentence, predictions)):
            if p_i == class_tag:
                if start_stop_words is None:
                    start_stop_words = i
                contains_stop_word = True
            elif w in STOP_WORDS and start_stop_words is None:
                start_stop_words = i
            elif w not in STOP_WORDS and start_stop_words is not None:
                if contains_stop_word:
                    stop_word_ranges.append((start_stop_words, i))
                start_stop_words = None
                contains_stop_word = False
        if contains_stop_word and start_stop_words is not None:
            stop_word_ranges.append((start_stop_words, len(predictions)))
        for (start_pos, end_pos) in stop_word_ranges:
            for i in range(start_pos, end_pos):
                proc_pred[i] = class_tag
        return proc_pred
    
    def convert_to_bio(self, sentence: List[str], predictions: List[str], class_tag: str) -> List[str]:
        proc_pred = list(predictions) # create copy
        for i, (w, p_i) in enumerate(zip(sentence, predictions)):
            if i == 0:
                if p_i == class_tag:
                    proc_pred[i] = f'B-{class_tag}'
                continue
            
            if p_i == class_tag:
                prev_tag = predictions[i - 1]
                if prev_tag == 'O':
                    proc_pred[i] = f'B-{class_tag}'
                else:
                    proc_pred[i] = f'I-{class_tag}'
        return proc_pred
    
    def convert(
        self,
        annotated_corpus: AnnotatedDataType,
    ) -> AnnotatedDataType:
        '''
        convert the annotations to BIO

        input:
            - annotated_corpus ``AnnotatedDataType``
                the annotated corpus to be converted to BIO encoding
        output:
            - annotations are in BIO format
        '''
        annotated_data = []
        for data_entry in annotated_corpus:
            data_entry = data_entry.copy()
            heuristic_output = self.stop_word_heuristic(
                sentence=data_entry['input'],
                predictions=data_entry['output'],
                class_tag=self.binary_class,
            )
            data_entry['output'] = self.convert_to_bio(
                data_entry['input'],
                heuristic_output,
                self.binary_class,
            )
            annotated_data.append(data_entry)
        return annotated_data