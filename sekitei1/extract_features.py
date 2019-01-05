
# coding: utf-8

# In[75]:

import sys
import re
import random
from operator import itemgetter
from collections import Counter
from urllib import unquote


def extract_features(INPUT_FILE_1, INPUT_FILE_2, OUTPUT_FILE):
    features = Counter()
    f_1 = open(INPUT_FILE_1, 'r')
    f_2 = open(INPUT_FILE_2, 'r')
    f_out = open(OUTPUT_FILE, 'w')
    flag_c = False
    flag_d = False

    list_1 = []
    list_2 = []
    list_rows = []

    for row in f_1:
        list_1.append(row.strip())
    for row in f_2:
        list_2.append(row.strip())

    list_rows.extend(random.sample(list_1, 1000))
    list_rows.extend(random.sample(list_2, 1000))

    for row in list_rows:
        row = unquote(row)
        #print row
        temp = re.split('/', row)
        temp = temp[3:]
        if temp[-1] == "":
            temp.pop()
        #print temp
        if re.match('\?', temp[len(temp) - 1]):
            param = temp.pop()
            list_param = re.split('&', param)
            for item in list_param:
                s = "param:" + item
                features[s] += 1
                s = "param_names:" + re.split('=', item)[0]
                features[s] += 1
        s = "segments:" + str(len(temp))
        features[s] += 1
        for i in xrange(len(temp)):
            flag_d = False
            flag_c = False

            # 4a
            s = "segment_name_" + str(i) + ":" + temp[i]
            features[s] += 1

            # 4b
            if re.match('\d+$',temp[i]):
                s = "segment_[0-9]_" + str(i) + ":1"
                features[s] += 1

            # 4c
            if re.match('[^\d]+\d+[^\d]+$', temp[i]):
                s = "segment_substr[0-9]_" + str(i) + ":1"
                features[s] += 1
                flag_c = True

            # 4d
            ext = re.search('\.([^\.]+)$', temp[i])
            if ext:
                features['segment_ext_' + str(i) + ':' + ext.group(1)] += 1
                flag_d = True

            # 4e
            if flag_c and flag_d:
				features['segment_ext_substr[0-9]_' + str(i) + ':' + ext.group(1)] += 1

            # 4f
            features['segment_len_' + str(i) + ':' + str(len(temp[i]))] += 1

    for item in features.most_common():
        if item[1] > 100:
            f_out.write(item[0] + "\t" + str(item[1]) + "\n")






