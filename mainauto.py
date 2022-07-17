### This is the main file for testing the performance of SFGAE.

import matplotlib.pyplot as plt
import mxnet as mx
import numpy as np
from scipy import interp
from sklearn import metrics
import warnings

from trainauto import Train


if __name__ == '__main__':
  warnings.filterwarnings("ignore")
  auc, acc, pre, recall, f1, fprs, tprs,aupr = Train(directory='data',
                                                epochs=100,
                                                aggregator='GraphSAGE',  # 'GraphSAGE'
                                                embedding_size=256,
                                                layers=2,
                                                dropout=0.7,
                                                slope=0.2,  # LeakyReLU
                                                lr=0.001,
                                                wd=2e-3,
                                                random_seed=126,  
                                                ctx=mx.cpu(0)) #GPU version: ctx=mx.gpu(0)

  print('seed: %.4f \n' % (126),
        '-AUC mean: %.4f, variance: %.4f \n' % (np.mean(auc), np.std(auc)),
        'Accuracy mean: %.4f, variance: %.4f \n' % (np.mean(acc), np.std(acc)), 
        'Precision mean: %.4f, variance: %.4f \n' % (np.mean(pre), np.std(pre)),
        'Recall mean: %.4f, variance: %.4f \n' % (np.mean(recall), np.std(recall)),
        'F1-score mean: %.4f, variance: %.4f \n' % (np.mean(f1), np.std(f1)),
        'aupr mean: %.4f, variance: %.4f \n' % (np.mean(aupr), np.std(aupr)),)
