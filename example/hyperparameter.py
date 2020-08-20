"""
메모장
아무말 대잔치
"""

import os

hParam = dict()

# general
hParam['data_path'] = os.path.join(os.getenv('HOME'), 'dataset/mydata/animals.txt')
hParam['model_path'] = os.path.join(os.getenv('HOME'), 'result/mymodel/model.param')
hParam['log_path'] = os.path.join(os.getenv('HOME'), 'result/mymodel/log.pickle')
hParam['verbose'] = True

# data
hParam['option'] = True

# model
hParam['inDim'] = 16
hParam['hiddenDim'] = 32
hParam['outDim'] = 10
hParam['numLayers'] = 4

# train
hParam['lr'] = 0.001
hParam['epoch'] = 10
hParam['batch'] = 1024
