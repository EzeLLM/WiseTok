#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys
import re
from collections import Counter
from math import log

stopwords_en = [u'the', u'of', u'and', u'in', u'a', u'to', u'was', u'is', u'for', u'as',\
                    u'on', u'by', u'with', u'he', u'at', u'that', u'from', u'his', u'it', u'an']
stopwords_ru = [u'в', u'и', u'на', u'с', u'по', u'года', u'из', u'году', u'не', u'к', u'а',\
                    u'от', u'для', u'был', u'что', u'его', u'как', u'за', u'также', u'до']
num_ru = 1236325
num_en = 4268154

re_en = re.compile(ur"[a-z]+")
re_ru = re.compile(ur"[а-яё]+")


class Mapper:

    def __init__(self, lang):
        if lang == 'ru':
            self.stopwords = stopwords_ru
            self.num_of_docs = num_ru
            self.re = re_ru
        else:
            self.stopwords = stopwords_en
            self.num_of_docs = num_en
            self.re = re_en


    def run(self):
        word_counter = Counter()

        data = self.readInput()

        for docid, contents in data:
            num_of_words = 0
            for match in self.re.finditer(contents):
                word = match.group(0)
                num_of_words += 1
                if word not in self.stopwords:
                    word_counter[word] += 1

            for word, count in word_counter.iteritems():
                print ('%s+%s+%lf' % (word, docid, float(word_counter[word])/float(num_of_words))).encode("utf-8")
            word_counter.clear()

    def readInput(self):
        for line in sys.stdin:
            yield unicode(line, 'utf8').lower().strip().split('\t', 1)


class Reducer:


    def __init__(self, lang):
        if lang == 'ru':
            self.stopwords = stopwords_ru
            self.num_of_docs = num_ru
        elif lang == 'en':
            self.stopwords = stopwords_en
            self.num_of_docs = num_en

    def run(self):
        data = self.readInput()

        cur_word = None
        cur_docs = 0
        top20 = []

        for element, elemid, elemfreq in data:
            word = element
            docid = int(elemid)
            word_freq = float(elemfreq)

            if cur_word == word:
                cur_docs += 1
                if len(top20) <= 20:
                    top20.append([docid, word_freq])
            else:
                if cur_word:
                    itf = log(float(self.num_of_docs)/float(cur_docs))
                    output = cur_word
                    for elem in top20:
                        output += '\t' + str(elem[0]) + ':' + str(elem[1] * itf)
                    print (output).encode("utf-8")
                cur_word = word
                top20 = [[docid, word_freq]]
                cur_docs = 1

        itf = log(float(self.num_of_docs)/float(cur_docs))
        output = cur_word
        for elem in top20:
            output += '\t' + str(elem[0]) + ':' + str(elem[1] * itf)
        print (output).encode("utf-8")

    def readInput(self):
        for line in sys.stdin:
            yield unicode(line, 'utf8').lower().strip().split('+')


if __name__ == "__main__":
    mr_func = sys.argv[1]
    lang = sys.argv[2]
    if mr_func == "map":
        mapper = Mapper(lang)
        mapper.run()
    elif mr_func == "reduce":
        reducer = Reducer(lang)
        reducer.run()