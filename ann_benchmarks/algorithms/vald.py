from __future__ import absolute_import

import atexit
import subprocess
import urllib.error
import urllib.request

import grpc
import yaml
from ann_benchmarks.algorithms.base import BaseANN

from vald import agent_pb2_grpc, payload_pb2

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
        'health_check_servers': [
            {
                'name': 'readiness',
                'host': '127.0.0.1',
                'port': 3001,
                'mode': '',
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
        'startup_strategy': ['agent-grpc', 'readiness'],
        'shutdown_strategy': ['readiness', 'agent-grpc'],
        'full_shutdown_duration': '600s',
        'tls': {
            'enabled': False,
        }
    },
    'ngt': {
        'enable_in_memory_mode': True
    }
}

grpc_opts = [
    ('grpc.keepalive_time_ms', 1000 * 10),
    ('grpc.keepalive_timeout_ms', 1000 * 10),
    ('grpc.max_connection_idle_ms', 1000 * 50)
]

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
        self._address = 'localhost:8082'

    def fit(self, X):
        dim = len(X[0])
        self._ngt_config['dimension'] = dim
        self._param['ngt'].update(self._ngt_config)
        with open('config.yaml', 'w') as f:
            yaml.dump(self._param, f)
        vectors = [
            payload_pb2.Object.Vector(
                id=str(i),
                vector=X[i].tolist()) for i in range(
                len(X))]

        p = subprocess.Popen(['/go/bin/ngt', '-f', 'config.yaml'])
        atexit.register(lambda: p.kill())
        while True:
            try:
                with urllib.request.urlopen('http://localhost:3001/readiness') as response:
                    if response.getcode() == 200:
                        break
            except (urllib.error.HTTPError, urllib.error.URLError):
                pass

        channel = grpc.insecure_channel(self._address, grpc_opts)
        stub = agent_pb2_grpc.AgentStub(channel)
        for _ in stub.StreamInsert(iter(vectors)):
            pass
        stub.CreateIndex(
            payload_pb2.Control.CreateIndexRequest(
                pool_size=10000))

    def set_query_arguments(self, epsilon):
        self._epsilon = epsilon - 1.0
        channel = grpc.insecure_channel(self._address, grpc_opts)
        self._stub = agent_pb2_grpc.AgentStub(channel)

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
