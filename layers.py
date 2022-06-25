import math
import mxnet as mx
from mxnet import ndarray as nd
from mxnet.gluon import nn
import dgl
import dgl.function as FN
import numpy as np

### gnn layer of SFGAE
class GraphSageLayer(nn.Block):
    def __init__(self, feature_size, G, disease_nodes, mirna_nodes, dropout, slope, ctx):
        super(GraphSageLayer, self).__init__()

        self.feature_size = feature_size
        self.G = G
        self.disease_nodes = disease_nodes
        self.mirna_nodes = mirna_nodes
        self.ctx = ctx

        self.disease_update1 = NodeUpdate1(feature_size, dropout, slope)
        self.miran_update1 = NodeUpdate1(feature_size, dropout, slope)

        self.disease_update = NodeUpdate(feature_size, dropout, slope)
        self.miran_update = NodeUpdate(feature_size, dropout, slope)

        all_nodes = mx.nd.arange(G.number_of_nodes(), dtype=np.int64)
        self.deg = G.in_degrees(all_nodes).astype(np.float32).copyto(ctx)

    def forward(self, G):
        assert G.number_of_nodes() == self.G.number_of_nodes()
        G.ndata['deg'] = self.deg

        G.update_all(FN.copy_src('h', 'h'), FN.sum('h', 'h_agg'))  # mean, max, sum


        G.apply_nodes(self.disease_update1, self.disease_nodes)
        G.apply_nodes(self.miran_update1, self.mirna_nodes)

        G.apply_nodes(self.disease_update, self.disease_nodes)
        G.apply_nodes(self.miran_update, self.mirna_nodes)

### gnn layer of SFGAE 
class GraphSageLayer0(nn.Block):
    def __init__(self, feature_size, G, disease_nodes, mirna_nodes, dropout, slope, ctx):
        super(GraphSageLayer0, self).__init__()

        self.feature_size = feature_size
        self.G = G
        self.disease_nodes = disease_nodes
        self.mirna_nodes = mirna_nodes
        self.ctx = ctx

        self.disease_update1 = NodeUpdate1(feature_size, dropout, slope)
        self.miran_update1 = NodeUpdate1(feature_size, dropout, slope)

        self.disease_update = NodeUpdate0(feature_size, dropout, slope)
        self.miran_update = NodeUpdate0(feature_size, dropout, slope)

        all_nodes = mx.nd.arange(G.number_of_nodes(), dtype=np.int64)
        self.deg = G.in_degrees(all_nodes).astype(np.float32).copyto(ctx)

    def forward(self, G):
        assert G.number_of_nodes() == self.G.number_of_nodes()
        G.ndata['deg'] = self.deg

        G.update_all(FN.copy_src('h', 'h'), FN.sum('h', 'h_agg'))  # mean, max, sum
        G.update_all(FN.copy_src('h1', 'h1'), FN.sum('h1', 'h1_agg')) 


        G.apply_nodes(self.disease_update1, self.disease_nodes)
        G.apply_nodes(self.miran_update1, self.mirna_nodes)


        G.apply_nodes(self.disease_update, self.disease_nodes)
        G.apply_nodes(self.miran_update, self.mirna_nodes)


### hidden embedding update
class NodeUpdate0(nn.Block):
    def __init__(self, feature_size, dropout, slope):
        super(NodeUpdate0, self).__init__()

        self.feature_size = feature_size
        self.leakyrelu = nn.LeakyReLU(slope)
        self.W = nn.Dense(feature_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.Activation('tanh')
        with self.name_scope():
            self.W_att = self.params.get('dot_weights_att1', shape=(feature_size, feature_size))
            self.V_att = self.params.get('dot_weights_att', shape=(feature_size, 1))

    def forward(self, nodes):

        h = nodes.data['h']
        h1 = nodes.data['h1']

        deg = nodes.data['deg'].expand_dims(1)

        h_agg = nodes.data['h_agg'] / nd.maximum(deg, 1e-5)
        h1_agg = nodes.data['h1_agg'] / nd.maximum(deg, 1e-5)

        h_att = nd.exp(nd.dot(self.activation(nd.dot(h_agg, self.W_att.data())),self.V_att.data()))
        hh_att = nd.exp(nd.dot(self.activation(nd.dot(h, self.W_att.data())),self.V_att.data()))
        h1_att = nd.exp(nd.dot(self.activation(nd.dot(h1_agg, self.W_att.data())),self.V_att.data()))  
        h_zongatt = h_att+h1_att+hh_att

        h_att = h_att/nd.maximum(h_zongatt, 1e-5)
        h_agg = h_att*h_agg

        h1_att = h1_att/nd.maximum(h_zongatt, 1e-5)
        h1_agg = h1_att*h1_agg

        hh_att = hh_att/nd.maximum(h_zongatt, 1e-5)
        hh_agg = hh_att*h

        
        h_concat = nd.concat(h, h_agg+h1_agg+hh_agg, dim=1)
  

        h_new = self.dropout(self.leakyrelu(self.W(h_concat)))

        return {'h': h_new}





class NodeUpdate1(nn.Block):
    def __init__(self, feature_size, dropout, slope):
        super(NodeUpdate1, self).__init__()

        self.feature_size = feature_size
        self.leakyrelu = nn.LeakyReLU(slope)
        self.W1 = nn.Dense(feature_size)
        self.dropout = nn.Dropout(0.7)

    def forward(self, nodes):
        h1 = nodes.data['h1']
        h1_new = self.dropout(self.leakyrelu(self.W1(h1)))
        return {'h1': h1_new}
