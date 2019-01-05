# -*- coding: utf-8 -*-
import pickle
from array import array
import mmh3
import sys

dictionary = {}
chunk_num = 0
offset = 0

with open('index.bin', 'wb') as bin_index:
    while True:
        try:
            index_chunk = pickle.load(open('index{}.pkl'.format(chunk_num), 'rb'))
        except:
            break

        for key in index_chunk:
            key_len = len(index_chunk[key])
            dictionary[mmh3.hash64(key.encode('utf8'))[0]] = [offset, key_len]
            offset += key_len
            bin_index.write(index_chunk[key])

        chunk_num += 1

module = 10000
list_hash = []
bin_dictionary = array('l', [module])

sorted_dict = sorted(dictionary.items())

hash_list = [array('l') for i in range(module)]

for key, val in sorted_dict:
    hash_num = key % module
    hash_list[hash_num].extend([key] + val)

cur_pos = 2 * module + 1

for i in xrange(module):
    dif = hash_list[i].buffer_info()[1]
    bin_dictionary.extend([cur_pos, dif / 3])
    cur_pos += dif

for i in xrange(module):
    bin_dictionary.extend(hash_list[i])

bin_dictionary.tofile(open('dictionary.bin', 'wb'))
