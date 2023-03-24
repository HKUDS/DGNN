import dgl, math, torch
import numpy as np
import networkx as nx
import torch.nn as nn
import dgl.function as fn


class HGMN(nn.Module):
    def __init__(self, args, n_user, n_item, n_category):
        super(HGMN, self).__init__()
        self.n_user = n_user
        self.n_item = n_item
        self.n_category = n_category

        self.n_hid = args.n_hid
        self.n_layers = args.n_layers
        self.mem_size = args.mem_size

        self.emb = nn.Parameter(torch.empty(n_user + n_item + n_category, self.n_hid))
        self.norm = nn.LayerNorm((args.n_layers + 1) * self.n_hid)

        self.layers = nn.ModuleList()
        for i in range(0, self.n_layers):
            self.layers.append(GNNLayer(self.n_hid, self.n_hid, self.mem_size, 5,
                                        layer_norm=True, dropout=args.dropout,
                                        activation=nn.LeakyReLU(0.2, inplace=True)))

        self.pool = GraphPooling('mean')

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.emb)

    def predict(self, user, item):
        return torch.einsum('bc, bc -> b', user, item) / user.shape[1]

    def forward(self, graph):
        x = self.emb

        all_emb = [x]
        for idx, layer in enumerate(self.layers):
            x = layer(graph, x)
            all_emb += [x]
        x = torch.cat(all_emb, dim=1)
        x = self.norm(x)

        # Pooling
        guu_es = graph.edata['type'] == 0
        graph_uu = dgl.graph((graph.edges()[0][guu_es], graph.edges()[1][guu_es]), num_nodes=self.n_user)
        graph_uu = dgl.add_self_loop(graph_uu)
        user_pool = self.pool(graph_uu, x[:self.n_user])

        return x, user_pool


class GraphPooling(nn.Module):
    def __init__(self, pool_type):
        super(GraphPooling, self).__init__()
        self.pool_type = pool_type
        if pool_type == 'mean':
            self.reduce_func = fn.mean(msg='m', out='h')
        elif pool_type == 'max':
            self.reduce_func = fn.max(msg='m', out='h')
        elif pool_type == 'min':
            self.reduce_func = fn.min(msg='m', out='h')

    def forward(self, g, feat):
        with g.local_scope():
            g.ndata['x'] = feat
            g.update_all(fn.copy_u('x', 'm'), self.reduce_func)
            return g.ndata['h']


class BPRLoss(nn.Module):
    def __init__(self, lamb_reg):
        super(BPRLoss, self).__init__()
        self.lamb_reg = lamb_reg

    def forward(self, pos_preds, neg_preds, *reg_vars):
        batch_size = pos_preds.size(0)

        bpr_loss = -0.5 * (pos_preds - neg_preds).sigmoid().log().sum() / batch_size
        reg_loss = torch.tensor([0.], device=bpr_loss.device)
        for var in reg_vars:
            reg_loss += self.lamb_reg * 0.5 * var.pow(2).sum()
        reg_loss /= batch_size

        loss = bpr_loss + reg_loss

        return loss, [bpr_loss.item(), reg_loss.item()]


class MemoryEncoding(nn.Module):
    def __init__(self, in_feats, out_feats, mem_size):
        super(MemoryEncoding, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.mem_size = mem_size
        self.linear_coef = nn.Linear(in_feats, mem_size, bias=True)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.linear_w = nn.Linear(mem_size, out_feats * in_feats, bias=False)

    def get_weight(self, x):
        coef = self.linear_coef(x)
        if self.act is not None:
            coef = self.act(coef)
        w = self.linear_w(coef)
        w = w.view(-1, self.out_feats, self.in_feats)
        return w

    def forward(self, h_dst, h_src):
        w = self.get_weight(h_dst)
        res = torch.einsum('boi, bi -> bo', w, h_src)
        return res


class GNNLayer(nn.Module):
    def __init__(self,
                in_feats,
                out_feats,
                mem_size,
                num_rels,
                bias=True,
                activation=None,
                self_loop=True,
                dropout=0.0,
                layer_norm=False):
        super(GNNLayer, self).__init__()
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.mem_size = mem_size

        self.num_rels = num_rels
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop
        self.layer_norm = layer_norm

        self.node_ME = MemoryEncoding(in_feats, out_feats, mem_size)
        self.rel_ME = nn.ModuleList([
            MemoryEncoding(in_feats, out_feats, mem_size)
                for i in range(self.num_rels)
        ])

        if self.bias:
            self.h_bias = nn.Parameter(torch.empty(out_feats))
            nn.init.zeros_(self.h_bias)

        if self.layer_norm:
            self.layer_norm_weight = nn.LayerNorm(out_feats)

        self.dropout = nn.Dropout(dropout)

    def message_func1(self, edges):
        msg = torch.empty((edges.src['h'].shape[0], self.out_feats),
                           device=edges.src['h'].device)
        for etype in range(self.num_rels):
            loc = edges.data['type'] == etype
            if loc.sum() == 0:
                continue
            src = edges.src['h'][loc]
            dst = edges.dst['h'][loc]
            sub_msg = self.rel_ME[etype](dst, src)
            msg[loc] = sub_msg
        return {'m': msg}

    def forward(self, g, feat):
        with g.local_scope():
            g.ndata['h'] = feat

            g.update_all(self.message_func1, fn.mean(msg='m', out='h'))
            # g.update_all(self.message_func2, fn.mean(msg='m', out='h'))

            node_rep = g.ndata['h']
            if self.layer_norm:
                node_rep = self.layer_norm_weight(node_rep)
            if self.bias:
                node_rep = node_rep + self.h_bias
            if self.self_loop:
                h = self.node_ME(feat, feat)
                node_rep = node_rep + h
            if self.activation:
                node_rep = self.activation(node_rep)
            node_rep = self.dropout(node_rep)
            return node_rep
