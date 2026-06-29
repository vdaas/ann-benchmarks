from __future__ import absolute_import
import sys
import os
import ctypes
import subprocess
import time

# NGT 2.7.4 ライブラリを先にロードしてから ngtpy をインポートする
_NGT274_BIN = '/tmp/ngt274/bin'
_NGT274_LIB = '/tmp/ngt274/lib'
_libngt274 = os.path.join(_NGT274_LIB, 'libngt.so')
if os.path.exists(_libngt274):
    ctypes.CDLL(_libngt274, ctypes.RTLD_GLOBAL)

import ngtpy  # noqa: E402
import numpy as np
from ann_benchmarks.algorithms.base import BaseANN
from ann_benchmarks.constants import INDEX_DIR

def _ngt_env():
    env = os.environ.copy()
    ld = env.get('LD_LIBRARY_PATH', '')
    if _NGT274_LIB not in ld:
        env['LD_LIBRARY_PATH'] = _NGT274_LIB + (':' + ld if ld else '')
    return env

class QG(BaseANN):
    def __init__(self, metric, object_type, epsilon, param):
        metrics = {'euclidean': '2', 'angular': 'E'}
        self._edge_size = int(param['edge'])
        self._outdegree = int(param['outdegree'])
        self._indegree = int(param['indegree'])
        self._max_edge_size = int(param['max_edge']) if 'max_edge' in param.keys() else 128
        self._metric = metrics[metric]
        self._object_type = object_type
        self._edge_size_for_search = int(param['search_edge']) if 'search_edge' in param.keys() else -2
        self._tree_disabled = (param['tree'] == False) if 'tree' in param.keys() else False
        self._build_time_limit = 4
        self._epsilon = epsilon
        print('QG: edge_size=' + str(self._edge_size))
        print('QG: outdegree=' + str(self._outdegree))
        print('QG: indegree=' + str(self._indegree))
        print('QG: edge_size_for_search=' + str(self._edge_size_for_search))
        print('QG: epsilon=' + str(self._epsilon))
        print('QG: metric=' + metric)
        print('QG: object_type=' + object_type)

    def fit(self, X):
        print('QG: start indexing...')
        dim = len(X[0])
        n = len(X)
        print('QG: # of data=' + str(n))
        print('QG: dimensionality=' + str(dim))
        tag = 'ngt-qg-D{}-N{}-E{}-od{}-id{}-me{}'.format(
            dim, n, self._edge_size, self._outdegree, self._indegree, self._max_edge_size)
        index_dir = os.path.join(INDEX_DIR, tag)
        os.makedirs(index_dir, exist_ok=True)
        index = os.path.join(
            index_dir,
            'ONNG-{}-{}-{}'.format(self._edge_size, self._outdegree,
                                   self._indegree))
        anngIndex = os.path.join(index_dir, 'ANNG-' + str(self._edge_size))
        print('QG: index=' + index)
        if (not os.path.exists(index)) and (not os.path.exists(anngIndex)):
            print('QG: create ANNG')
            t = time.time()
            ngt_bin = os.path.join(_NGT274_BIN, 'ngt')
            args = [ngt_bin, 'create', '-it', '-p8', '-b500', '-ga', '-of',
                    '-D' + self._metric, '-d' + str(dim),
                    '-E' + str(self._edge_size), '-S40',
                    '-e' + str(self._epsilon), '-P0', '-B30',
                    '-T' + str(self._build_time_limit), anngIndex]
            subprocess.call(args, env=_ngt_env())
            idx = ngtpy.Index(path=anngIndex)
            idx.batch_insert(X, num_threads=24, debug=False)
            idx.save()
            idx.close()
            print('QG: ANNG construction time(sec)=' + str(time.time() - t))
        if not os.path.exists(index):
            print('QG: degree adjustment')
            t = time.time()
            ngt_bin = os.path.join(_NGT274_BIN, 'ngt')
            args = [ngt_bin, 'reconstruct-graph', '-mS',
                    '-E ' + str(self._outdegree),
                    '-o ' + str(self._outdegree),
                    '-i ' + str(self._indegree), anngIndex, index]
            subprocess.call(args, env=_ngt_env())
            print('QG: degree adjustment time(sec)=' + str(time.time() - t))
        if not os.path.exists(index + '/qg'):
            print('QG: quantization')
            t = time.time()
            ngtpy.QuantizedIndex.quantize(index, dimension_of_subvector=0,
                                          max_num_of_edges=self._max_edge_size)
            print('QG: quantization time(sec)=' + str(time.time() - t))
        if os.path.exists(index):
            print('QG: index already exists! ' + str(index))
            t = time.time()
            self.index = ngtpy.QuantizedIndex(index, self._max_edge_size)
            self.index.set_with_distance(False)
            self.indexName = index
            print('QG: open time(sec)=' + str(time.time() - t))
        else:
            print('QG: something wrong.')
        print('QG: end of fit')

    def set_query_arguments(self, parameters):
        result_expansion, epsilon = parameters
        print("QG: result_expansion=" + str(result_expansion))
        print("QG: epsilon=" + str(epsilon))
        self.name = 'QG-NGT(%s, %s, %s, %s, %s, %1.3f)' % (
            self._edge_size, self._outdegree,
            self._indegree, self._max_edge_size,
            epsilon,
            result_expansion)
        epsilon = epsilon - 1.0
        self.index.set(epsilon=epsilon, result_expansion=result_expansion)

    def query(self, v, n):
        return self.index.search(v, n)

    def freeIndex(self):
        print('QG: free')
