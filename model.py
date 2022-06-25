### In this file, we construct SFGAE model, which consists of encoder step and decoder step.

import mxnet as mx
from mxnet import ndarray as nd
from mxnet.gluon import nn
import dgl
import numpy as np

from layers import GraphSageLayer,GraphSageLayer0

####  SFGAE model class ####
class SFGAE(nn.Block):
    def __init__(self, encoder,decoder):
    # def __init__(self, encoder,decoder_FM):
    # def __init__(self, encoder, decoder,decoder_FM):
        super(SFGAE, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        # self.decoder_FM = decoder_FM

    def forward(self, G, diseases, mirnas):
        h = self.encoder(G)   ####  gnn encoder module  ####
        h_diseases = h[diseases]    ####  disease node  feature  ####
        h_mirnas = h[mirnas]        ####  miRNA node  feature  ####
        out2 = self.decoder(h_diseases, h_mirnas)   ####  sfgae decoder module  ####
        return out2


class GraphEncoder(nn.Block):
    def __init__(self, embedding_size, n_layers, G, aggregator, dropout, slope, ctx):
        super(GraphEncoder, self).__init__()

        self.G = G
        self.disease_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 1).astype(np.int64).copyto(ctx)
        self.mirna_nodes = G.filter_nodes(lambda nodes: nodes.data['type'] == 0).astype(np.int64).copyto(ctx)


        self.layers = nn.Sequential()
        for i in range(n_layers):
            if i >= n_layers-8:
                self.layers.add(
                    GraphSageLayer0(embedding_size, G, self.disease_nodes, self.mirna_nodes, dropout, slope, ctx))
            else:
                if aggregator == 'GraphSAGE':

                    self.layers.add(
                        GraphSageLayer(embedding_size, G, self.disease_nodes, self.mirna_nodes, dropout, slope, ctx))
                else:
                    raise NotImplementedError

        self.disease_emb = DiseaseEmbedding(embedding_size, dropout)
        self.mirna_emb = MirnaEmbedding(embedding_size, dropout)

    def forward(self, G):
        #### Generate embedding on disease nodes and mirna nodes ####
        assert G.number_of_nodes() == self.G.number_of_nodes()

        G.apply_nodes(lambda nodes: {'h': self.disease_emb(nodes.data)}, self.disease_nodes)
        G.apply_nodes(lambda nodes: {'h': self.mirna_emb(nodes.data)}, self.mirna_nodes)

        G.apply_nodes(lambda nodes: {'h1': self.disease_emb(nodes.data)}, self.disease_nodes)
        G.apply_nodes(lambda nodes: {'h1': self.mirna_emb(nodes.data)}, self.mirna_nodes)


        for layer in self.layers:
            layer(G)
        print(G.ndata['h'])
        print(G.ndata['h1'])

        return nd.concat(G.ndata['h'], G.ndata['h1'], dim=1)


class DiseaseEmbedding(nn.Block):
    def __init__(self, embedding_size, dropout):
        super(DiseaseEmbedding, self).__init__()

        seq = nn.Sequential()
        with seq.name_scope():
            seq.add(nn.Dense(embedding_size))
            # seq.add(nn.Activation('relu'))
            seq.add(nn.Dropout(dropout))
        self.proj_disease = seq
        
    def forward(self, ndata):
        #### Generate h0 of disease nodes####
        extra_repr = self.proj_disease(ndata['d_features'])

        return extra_repr


class MirnaEmbedding(nn.Block):
    def __init__(self, embedding_size, dropout):
        super(MirnaEmbedding, self).__init__()

        seqm = nn.Sequential()
        with seqm.name_scope():
            seqm.add(nn.Dense(embedding_size))
            seqm.add(nn.Dropout(dropout))
        self.proj_mirna = seqm

    def forward(self, ndata):
        #### Generate h0 of miRNA nodes####
        extra_repr = self.proj_mirna(ndata['m_features'])
        return extra_repr


class BilinearDecoder(nn.Block):
    def __init__(self, feature_size):
        super(BilinearDecoder, self).__init__()

        self.activation = nn.Activation('sigmoid')
        with self.name_scope():
            self.W = self.params.get('dot_weights', shape=(2*feature_size, 2*feature_size))

    def forward(self, h_diseases, h_mirnas):
        results_mask = self.activation((nd.dot(h_diseases, self.W.data()) * h_mirnas).sum(1))

        return results_mask




class BilinearDecoder_FM(nn.Block):
    def __init__(self, feature_size):
        super(BilinearDecoder_FM, self).__init__()

        self.activation = nn.Activation('sigmoid')
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.Wout1 = nn.Dense(feature_size, use_bias=False)
        self.Wout2 = nn.Dense(feature_size, use_bias=False)
        self.dropout = nn.Dropout(0.7)
        with self.name_scope():
            self.WV = self.params.get('dot_weightsV', shape=(feature_size, 2*feature_size))
            self.WX = self.params.get('dot_weightsX', shape=(feature_size,1))

    def forward(self, h_diseases, h_mirnas):
        h_mlpconcat = nd.concat(h_diseases, h_mirnas, dim=1)
        h_mlpconcat = self.dropout(self.leakyrelu(self.Wout1(h_mlpconcat)))
        h_mlpconcat = self.dropout(self.leakyrelu(self.Wout2(h_mlpconcat)))
        part1 = nd.dot(h_mlpconcat, self.WX.data()).reshape(-1)
        part2_0 = nd.dot(h_mlpconcat, self.WV.data())
        part2_1 = nd.dot(h_mlpconcat*h_mlpconcat, self.WV.data()*self.WV.data())

        part2 = (part2_0 * part2_0).sum(1)-part2_1.sum(1)
        results_mask = self.activation(part1+part2/2)

        return results_mask
