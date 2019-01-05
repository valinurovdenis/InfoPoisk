#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This just a draft for homework 'near-duplicates'
Use MinshinglesCounter to make result closer to checker
"""

import re
import string
import numpy as np

import mmh3

from docreader import *


class TextNormalizer:
    @staticmethod
    def join_numbers(text):
        regex = re.compile('([\d])[\s]+([\d])')
        return regex.sub('\\1\\2', text)

    @staticmethod
    def clean_out_punct(text):
        regex = re.compile('[%s]' % re.escape(string.punctuation + "«" + "»"))
        return regex.sub(' ', text)

    @staticmethod
    def lower_case(text):
        return text.lower()

    @staticmethod
    def remove_entities(text):
        regex = re.compile('&[0-9a-z_A-Z]+;')
        return regex.sub(' ', text)


class MinshinglesCounter:
    SPLIT_RGX = re.compile(r'\w+', re.U)

    def __init__(self, window=5, n=20):
        self.window = window
        self.n = n

    def count(self, text):
        words = MinshinglesCounter._extract_words(text)
        shs = self._count_shingles(words)
        mshs = self._select_minshingles(shs)

        return None if None in mshs else mshs

    def _get_order_function(self, num):
        basis = [1663, 1999, 2203, 2381, 2411,
                 2657, 2789, 2843, 2861, 2909,
                 2953, 3169, 3217, 3259, 3491,
                 3467, 3469, 3499, 3511, 27644437]
        return lambda x: divmod(x, basis[num])[1]

    def _select_minshingles(self, shs):
        minshingle = [None]*self.n

        for i in range(self.n):
            h_i = self._get_order_function(i)
            for sh in shs:
                if minshingle[i] is None or h_i(sh) < h_i(minshingle[i]):
                    minshingle[i] = sh

        return minshingle

    def _count_shingles(self, words):
        shingles = []
        for i in xrange(len(words) - self.window):
            h = mmh3.hash(' '.join(words[i:i+self.window]).encode('utf-8'))
            shingles.append(h)
        return shingles

    @staticmethod
    def _extract_words(text):
        words = re.findall(MinshinglesCounter.SPLIT_RGX, text)
        return words


def main():
    files = sys.argv[1:] # all input files in current test case
    minshingles_counter = MinshinglesCounter(window=5, n=20)
    doc_urls = []
    minshingles = []

    reader = DocumentStreamReader(files)
    for doc in reader:
        text = doc.text
        text = TextNormalizer.join_numbers(text) # replace '40 000' with '40000'
        text = TextNormalizer.remove_entities(text) # replace '&nbsp;', '&amp;' and other entities with whitespace
        text = TextNormalizer.clean_out_punct(text) # remove punctuation
        text = TextNormalizer.lower_case(text) # lower case of the whole text

        minshingles.append(np.array(minshingles_counter.count(text)))
        #print minshingles[-1]
        doc_urls.append(doc.url)

    for i in range(len(doc_urls)):
        for j in range(len(doc_urls)):
            if i < j:
                measure = float(np.count_nonzero(minshingles[i] == minshingles[j])) / 20
                if measure > 0.75:
                    print doc_urls[i], doc_urls[j], measure


if __name__ == '__main__':
    main()
