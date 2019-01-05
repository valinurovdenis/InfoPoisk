# coding: utf-8


import sys
import os
import re
import random
import time
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
import numpy as np
import collections
import urllib
from operator import itemgetter
import scipy.spatial.distance as pdist
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


from sklearn.base import ClusterMixin

class Kmeans:

   def __init__(self, k=2, metric='euclidean', max_iter=5000, random_state=None, init='k-means'):
       """
       Инициализация метода
       :k - количество кластеров
       :metric - функция расстояния между объектами
       :max_iter - максиальное количество итераций
       :random_state - seed для инициализации генератора случайных чисел
       """

       self.k = k
       self.random_state = random_state
       self.metric = metric
       self.max_iter = max_iter
       self.init = init

   def good_values_for_centroids(self, X):
       self.centroids = np.array([X[np.random.randint(X.shape[0])]])

       for i in xrange(self.k - 1):
           distances = pdist.cdist(X, self.centroids, metric = self.metric).min(1) ** 2
           new_centroid = X[np.random.choice(X.shape[0], 1, p = distances / distances.sum(0))]
           self.centroids = np.append(self.centroids, new_centroid, axis = 0)

   def fit(self, X, y=None):
       """
       Процедура обучения k-means
       """

       # Инициализация генератора случайных чисел
       np.random.seed(self.random_state)

       # Массив с метками кластеров для каждого объекта из X
       self.labels = np.empty(X.shape[0])

       if self.init == 'k-means':
           self.good_values_for_centroids(X)
       else:
           self.centroids = X[np.random.choice(X.shape[0], self.k)]

       previous_centroids = np.empty((self.k, X.shape[1]))
       eps = 1e-12

       for i in xrange(self.max_iter):
           self.labels = pdist.cdist(X, self.centroids, metric = self.metric).argmin(1)
           for j in xrange(self.k):
               b = self.labels == j
               if np.count_nonzero(b == True) != 0:
                   self.centroids[j] = X[b].sum(0) / np.count_nonzero(b == True)
           if  np.absolute(self.centroids - previous_centroids).sum() < eps:
               break
           previous_centroids = np.copy(self.centroids)

       return self


   def predict(self, X, y=None):
       """
       Процедура предсказания кластера

       Возвращает метку ближайшего кластера для каждого объекта
       """

       return pdist.cdist(X, self.centroids, metric = self.metric).argmin(1)


def extract_features(ALL_URLS):
       features = collections.defaultdict(int)

       for url in ALL_URLS:
               url = urllib.unquote(url).strip()

               segments = re.split('/', url)
               segments = segments[3:]
               if segments[-1] == '':
                       segments.pop()

               if re.match('\?', segments[-1]):
                       segments[-1] = segments[-1][1:]
                       params = re.split('&', segments[-1])
                       for param in params:
                               param_name = re.split('=', param)[0]
                               features['param_name:' + param_name] += 1
                               features['param:' + param] += 1
                       segments.pop()

               features['segments:' + str(len(segments))] += 1

               for i, segment in enumerate(segments):
                       segment = segment.lower()

                       features['segment_name_' + str(i) + ':' + segment] += 1

                       if re.match('^\d+$', segment):
                               features['segment_[0-9]_' + str(i) + ":1"] += 1

                       if re.findall(':', segment):
                               features['colon_' + str(i)] += 1

                       if re.findall(',', segment):
                               features['zap_' + str(i)] += 1

                       if re.findall('\(', segment):
                               features['parenth_' + str(i)] += 1

                       if re.findall(' ', segment):
                               features['probel_' + str(i)] += 1

                       if re.findall('_', segment):
                               features['nizhnpod_' + str(i)] += 1

                       if re.findall('[a-z]', segment):
                               features['engla' + str(i)] += 1


                       c, d = False, False
                       if re.match('[^\d]+\d+[^\d]+$', segment):
                               features['segment_substr[0-9]_' + str(i) + ':1'] += 1
                               c = True

                       ext = re.search('\.([^\.]+)$', segment)
                       if ext:
                               features['segment_ext_' + str(i) + ':' + ext.group(1)] += 1
                               d = True

                       if c and d:
                               features['segment_ext_substr[0-9]_' + str(i) + ':' + ext.group(1)] += 1

                       features['segment_len_' + str(i) + ':' + str(len(segment))] += 1


       ret = dict()
       num = 0
       for feature, cnt in sorted(features.items(), key=itemgetter(1), reverse=True):
               if cnt < 100:
                       break
               ret[feature] = num
               num += 1

       return ret

def get_url_features(url):
   url = urllib.unquote(url).strip()

   segments = re.split('/', url)
   segments = segments[3:]
   if segments[-1] == '':
           segments.pop()
   features = []
   if re.match('\?', segments[-1]):
           segments[-1] = segments[-1][1:]
           params = re.split('&', segments[-1])
           for param in params:
                   param_name = re.split('=', param)[0]
                   features.append('param_name:' + param_name)
                   features.append('param:' + param)
           segments.pop()

   features.append('segments:' + str(len(segments)))

   for i, segment in enumerate(segments):
           segment = segment.lower()

           features.append('segment_name_' + str(i) + ':' + segment)

           if re.match('^\d+$', segment):
                   features.append('segment_[0-9]_' + str(i) + ":1")

           if re.findall(':', segment):
                   features.append('colon_' + str(i))

           if re.findall(',', segment):
                   features.append('zap_' + str(i))

           if re.findall('[a-z]', segment):
                   features.append('engla' + str(i))

           if re.findall('\(', segment):
                   features.append('parenth_' + str(i))


           if re.findall(' ', segment):
                   features.append('probel_' + str(i))

           if re.findall('_', segment):
                   features.append('nizhnpod_' + str(i))

           c, d = False, False
           if re.match('[^\d]+\d+[^\d]+$', segment):
                   features.append('segment_substr[0-9]_' + str(i) + ':1')
                   c = True

           ext = re.search('\.([^\.]+)$', segment)
           if ext:
                   features.append('segment_ext_' + str(i) + ':' + ext.group(1))
                   d = True

           if c and d:
                   features.append('segment_ext_substr[0-9]_' + str(i) + ':' + ext.group(1))

           features.append('segment_len_' + str(i) + ':' + str(len(segment)))

   return features


model = None
prob = None
all_features = None
qlink_model = None

def define_segments(QLINK_URLS, UNKNOWN_URLS, QUOTA):
   all_urls = QLINK_URLS + UNKNOWN_URLS
   global all_features
   global model
   global prob
   global qlink_model
   global use_prob

   all_features = extract_features(all_urls)

   X = np.zeros(shape=(len(all_urls), len(all_features)), dtype=int)

   for num, url in enumerate(all_urls):
       url_features = get_url_features(url)
       for name in url_features:
           if name in all_features:
               X[num, all_features[name]] = 1

   cluster_cnt = 10
   model = Kmeans(k=cluster_cnt, metric='jaccard', random_state=123).fit(X)
   #model = KMeans(n_clusters = cluster_cnt, random_state=123).fit(X)
   labels = model.predict(X)
   prob = np.zeros(cluster_cnt, dtype=float)
   use_prob = np.ones(cluster_cnt, dtype=float)

   min_quota = 400

   for i in range(cluster_cnt):
       temp = (labels == i)
       prob[i] = np.count_nonzero(temp[:len(QLINK_URLS)-1]) / float(len(QLINK_URLS)) * QUOTA
       if np.count_nonzero(temp) != 0 and np.count_nonzero(temp[len(QLINK_URLS):]) / float(np.count_nonzero(temp)) < 0.6:
           use_prob[i] = np.count_nonzero(temp[len(QLINK_URLS):]) / float(np.count_nonzero(temp)) * 0.8
       if np.count_nonzero(temp[:len(QLINK_URLS)-1]) / float(len(QLINK_URLS)) > 0.6:
           min_quota *= 5

   prob = np.maximum(min_quota, prob)

   y_qlink = np.zeros(len(all_urls), dtype=int)
   y_qlink[:len(QLINK_URLS)-1] = 1
   qlink_model = LogisticRegression().fit(X, y_qlink)

#
# returns True if need to fetch url
#



def fetch_url(url):
   #global sekitei
   #return sekitei.fetch_url(url);
   global qlink_fetched
   global not_qlink_fetched
   row = np.zeros(len(all_features))
   url_features = get_url_features(url)
   for name in url_features:
       if name in all_features:
           row[all_features[name]] = 1
   y = model.predict([row])[0]

   if qlink_model.predict([row])[0] == 1:
       prob[y] -= 1
       return True

   if prob[y] > 0 and np.random.random_sample() < use_prob[y]:
       prob[y] -= 1
       return True
   return False

