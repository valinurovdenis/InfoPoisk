# -*- coding: utf-8 -*-
import docreader
import doc2words
import cPickle as pickle
import sys
import mmh3
from array import array
from coder import Coder


def main(variant):
    with open('variant', 'w') as f:
        f.write(variant)
    
    encoder = Coder(variant)
    paths = []
    chunk_num = 0
    max_chunk_num = 2

    while True:
        tokens = {}
        i = 1
        if chunk_num == max_chunk_num:
            break     

        documents = docreader.DocumentStreamReader(docreader.parse_command_line().files)
        for doc in documents:
            if chunk_num == 0:
                paths.append(doc.url)

            words = doc2words.extract_words(doc.text)

            for word in set(words):
                if word in tokens:
                    tokens[word].append(i)
                elif len(word) % max_chunk_num == chunk_num:
                    tokens[word] = array('l', [i])

            i += 1

        for token in tokens:
            tokens[token] = encoder.encode(tokens[token])

        with open('index{}.pkl'.format(chunk_num), 'wb') as f:
            pickle.dump(tokens, f)

        chunk_num += 1
        first = False

    with open('paths.pkl', 'wb') as f:
        pickle.dump(paths, f)


if __name__ == '__main__':
    variant = sys.argv.pop(1)
    main(variant)
