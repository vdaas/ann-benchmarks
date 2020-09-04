from __future__ import absolute_import
import grpc
import numpy
from vald import payload_pb2
from vald import agent_pb2_grpc
from ann_benchmarks.algorithms.base import BaseANN
import yaml
import subprocess
import time
import atexit


default_server_config = {
    'version': 'v0.0.0',
    'logging': {
        'logger': 'glg',
        'level': 'info',
        'format': 'json'
    },
    'server_config': {
        'servers': [
            {
                'name': 'agent-grpc',
                'host': '127.0.0.1',
                'port': 8082,
                'mode': 'GRPC',
                'probe_wait_time': '3s',
                'http': {
                    'shutdown_duration': '5s',
                    'handler_timeout': '',
                    'idle_timeout': '',
                    'read_header_timeout': '',
                    'read_timeout': '',
                    'write_timeout': ''
                }
            }
        ],
        'startup_strategy': ['agent-grpc'],
        'shutdown_strategy': ['agent-grpc'],
        'full_shutdown_duration': '600s',
        'tls': {
            'enabled': False,
        }
    },
    'ngt': {
        'enable_in_memory_mode': True
    }
}

metrics = {'euclidean': 'l2', 'angular': 'cosine'}


class Vald(BaseANN):
    def __init__(self, metric, object_type, params):
        self._param = default_server_config
        self._ngt_config = {
            'distance_type': metrics[metric],
            'object_type': object_type,
            'search_edge_size': int(params['searchedge']),
            'creation_edge_size': int(params['edge']),
            'bulk_insert_chunk_size': int(params['bulk'])
        }
        channel = grpc.insecure_channel('localhost:8082')
        self._stub = agent_pb2_grpc.AgentStub(channel)

    def fit(self, X):
        self._ngt_config['dimension'] = len(X[0])
        self._param['ngt'].update(self._ngt_config)
        with open('config.yaml', 'w') as f:
            yaml.dump(self._param, f)
        vectors = [payload_pb2.Object.Vector(id=str(i), vector=X[i].tolist()) for i in range(len(X))]

        pid = subprocess.Popen(['/go/bin/ngt', '-f', 'config.yaml'])
        atexit.register(lambda: pid.kill())
        time.sleep(10)

        for _ in self._stub.StreamInsert(iter(vectors)): pass
        self._stub.CreateIndex(payload_pb2.Control.CreateIndexRequest(pool_size=10000))

    def set_query_arguments(self, epsilon):
        self._epsilon = epsilon - 1.0

    def query(self, v, n):
        response = self._stub.Search(payload_pb2.Search.Request(
            vector=v.tolist(),
            config=payload_pb2.Search.Config(
                num=n, radius=-1.0, epsilon=self._epsilon, timeout=3000000
            )))
        return [int(result.id) for result in response.results]

    def __str__(self):
        return 'Vald(%d, %d, %d, %1.3f)' % (
            self._ngt_config['creation_edge_size'],
            self._ngt_config['search_edge_size'],
            self._ngt_config['bulk_insert_chunk_size'],
            self._epsilon + 1.0
        )
