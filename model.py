import dgl
import numpy as np
import torch
from torch import nn
import dgl.nn as dglnn
import torch.nn as nn
import dgl.function as fn
import random
import tqdm
import sklearn.metrics
from torch import cosine_similarity
import torch.nn.functional as F
from dgl.nn.functional import edge_softmax
from dgl.utils import expand_as_pair
from dgl.nn.pytorch.utils import Identity
import dgl.function as fn
import torch as th
from torch import nn
from utils import *
from torch.nn import init


class Edge_level(nn.Module):
    def __init__(self,  in_feats,  out_feats,  num_heads, edge_dim=5,  feat_drop=0., attn_drop=0., negative_slope=0.2,residual=False, activation=None,allow_zero_in_degree=False, bias=True):
        super(Edge_level, self).__init__()
        self.edge_dim = edge_dim
        self._num_heads = num_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_heads, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_heads, bias=False)
        self.fc_edg = nn.Linear(self.edge_dim, self.edge_dim * num_heads)
        self.attn_l = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(th.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_edg = nn.Parameter(th.FloatTensor(size=(1, num_heads, self.edge_dim)))
        self.lin_out = nn.Linear(self.edge_dim + self._out_feats, self._out_feats)
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if bias:
            self.bias = nn.Parameter(th.FloatTensor(size=(num_heads * out_feats,)))
        else:
            self.register_buffer('bias', None)
        if residual:
            if self._in_dst_feats != out_feats * num_heads:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        # Reinitialize learnable parameters.
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
            nn.init.xavier_normal_(self.lin_out.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_edg, gain=gain)
        if self.bias is not None:
            nn.init.constant_(self.bias, 0)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, edge_fea,):
        with graph.local_scope():
            if isinstance(feat, tuple):
                src_prefix_shape = feat[0].shape[:-1]
                dst_prefix_shape = feat[1].shape[:-1]  #
                edge_fea_shape = edge_fea.shape[:-1]
                h_src = self.feat_drop(feat[0])
                h_dst = self.feat_drop(feat[1])
                edge_fea = self.feat_drop(edge_fea)
                if not hasattr(self, 'fc_src'):
                    feat_src = self.fc(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats)
                    feat_dst = self.fc(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats)
                    edge_fea = self.fc_edg(edge_fea).view(
                        *edge_fea_shape, self._num_heads, self.edge_dim)
                else:
                    feat_src = self.fc_src(h_src).view(
                        *src_prefix_shape, self._num_heads, self._out_feats)
                    feat_dst = self.fc_dst(h_dst).view(
                        *dst_prefix_shape, self._num_heads, self._out_feats)
            else:
                src_prefix_shape = dst_prefix_shape = feat.shape[:-1]
                h_src = h_dst = self.feat_drop(feat)
                feat_src = feat_dst = self.fc(h_src).view(
                    *src_prefix_shape, self._num_heads, self._out_feats)
                if graph.is_block:
                    feat_dst = feat_src[:graph.number_of_dst_nodes()]
                    h_dst = h_dst[:graph.number_of_dst_nodes()]
                    dst_prefix_shape = (graph.number_of_dst_nodes(),) + dst_prefix_shape[1:]

            el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
            er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
            e_e = (edge_fea * self.attn_edg).sum(dim=-1).unsqueeze(-1)
            graph.edata['_edge_weight'] = edge_fea
            graph.srcdata.update({'ft': feat_src, 'el': el})
            graph.dstdata.update({'er': er})
            graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
            data_e = graph.edata.pop('e')
            data_e = data_e + e_e
            e = self.leaky_relu(data_e)
            # compute softmax
            graph.edata['a'] = self.attn_drop(edge_softmax(graph,e))
            # message passing
            def edge_udf(edges):
                return {'he': torch.mul(edges.data['a'], edges.data['_edge_weight'])}

            graph.update_all(edge_udf, fn.sum('he', 'ft_f'))
            graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                             fn.sum('m', 'ft'))
            f = graph.dstdata['ft_f']
            rst = graph.dstdata['ft']
            rst = self.lin_out(torch.cat([f, rst], dim=-1))
            # residual
            if self.res_fc is not None:
                # Use -1 rather than self._num_heads to handle broadcasting
                resval = self.res_fc(h_dst).view(
                    *dst_prefix_shape, -1, self._out_feats)
                rst = rst + resval
            # bias
            if self.bias is not None:
                rst = rst + self.bias.view(
                    *((1,) * len(dst_prefix_shape)), self._num_heads, self._out_feats)
            # activation
            if self.activation:
                rst = self.activation(rst)
            return rst


class Semantic_level(nn.Module):
    def __init__(self, in_size, num_head, hidden_size=128):
        super(Semantic_level, self).__init__()
        self.Linear1 = nn.Linear(in_size * num_head, hidden_size)
        self.tanh = nn.Tanh()
        self.Linear2 = nn.Linear(hidden_size, 1, bias=False)
        self.num_head = num_head
        self.in_size = in_size
    def forward(self, z):
        z = th.stack(z, dim=0)
        z = z.transpose(1, 0, )
        z = th.reshape(z, (z.shape[0], z.shape[1], z.shape[2] * z.shape[3]))
        w = self.Linear1(z)
        w = self.tanh(w)
        w = self.Linear2(w).mean(0)
        beta = torch.softmax(w, dim=0)
        beta = beta.expand((z.shape[0],) + beta.shape)
        beta = (beta * z).sum(1)
        beta = th.reshape(beta, (beta.shape[0], self.num_head, self.in_size))
        return beta


class SparseSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, window_size=16, dilation=3, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.window_size = window_size
        self.dilation = dilation
        self.head_dim = d_model // num_heads
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5

    def _chunk_attention(self, x, chunk_size):
        # x: [Batch, Seq_Len, D_model]
        B, L, D = x.shape
        pad_len = (chunk_size - L % chunk_size) % chunk_size
        if pad_len > 0:
            x = F.pad(x, (0, 0, 0, pad_len))
        
        L_padded = x.shape[1]
        num_chunks = L_padded // chunk_size
        
        x_reshaped = x.view(B, num_chunks, chunk_size, D)
        x_features = x_reshaped.view(B * num_chunks, chunk_size, D)
        
        q = self.q_proj(x_features).view(B * num_chunks, chunk_size, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x_features).view(B * num_chunks, chunk_size, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x_features).view(B * num_chunks, chunk_size, self.num_heads, self.head_dim).transpose(1, 2)
        
        attn_weights = (q @ k.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        out = (attn_weights @ v).transpose(1, 2).reshape(B * num_chunks, chunk_size, D)
        out = out.view(B, num_chunks, chunk_size, D).view(B, L_padded, D)
        
        if pad_len > 0:
            out = out[:, :L, :]
            
        return out

    def forward(self, x):
        # 1. Local Attention
        local_out = self._chunk_attention(x, self.window_size)
        
        # 2. Atrous Attention
        B, L, D = x.shape
        pad_len = (self.dilation - L % self.dilation) % self.dilation
        if pad_len > 0:
            x_pad = F.pad(x, (0, 0, 0, pad_len))
        else:
            x_pad = x
            
        L_pad = x_pad.shape[1]
        # Reshape [B, L//r, r, D] -> [B*r, L//r, D]
        x_view = x_pad.view(B, L_pad // self.dilation, self.dilation, D)
        x_atrous_in = x_view.permute(0, 2, 1, 3).reshape(B * self.dilation, L_pad // self.dilation, D)
        
        atrous_out_temp = self._chunk_attention(x_atrous_in, self.window_size)
        
        atrous_out_temp = atrous_out_temp.view(B, self.dilation, L_pad // self.dilation, D).permute(0, 2, 1, 3).reshape(B, L_pad, D)
        
        if pad_len > 0:
            atrous_out = atrous_out_temp[:, :L, :]
        else:
            atrous_out = atrous_out_temp
            
        output = local_out + atrous_out
        return self.out_proj(output)

class SparseTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, window_size=16, dilation=4, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = SparseSelfAttention(d_model, num_heads, window_size, dilation, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, src):
        src2 = self.self_attn(src)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class HeteroGraph(nn.Module):
    def __init__(self, mods, in_size_sem, num_head):
        super(HeteroGraph, self).__init__()
        self.semantic_attention1 = Semantic_level(in_size=in_size_sem, num_head=num_head)
        self.semantic_attention2 = Semantic_level(in_size=in_size_sem, num_head=num_head)
        self.mods = nn.ModuleDict(mods)
        for _, v in self.mods.items():
            set_allow_zero_in_degree_fn = getattr(v, 'set_allow_zero_in_degree', None)
            if callable(set_allow_zero_in_degree_fn):
                set_allow_zero_in_degree_fn(True)

    def forward(self, g, inputs, edge_attr, mod_args=None, mod_kwargs=None):
        if mod_args is None:
            mod_args = {}
        if mod_kwargs is None:
            mod_kwargs = {}
        outputs = {nty: [] for nty in g.dsttypes}
        if isinstance(inputs, tuple) or g.is_block:
            if isinstance(inputs, tuple):
                src_inputs, dst_inputs = inputs
            else:
                src_inputs = inputs
                dst_inputs = {k: v[:g.number_of_dst_nodes(k)] for k, v in inputs.items()}

            for stype, etype, dtype in g.canonical_etypes:
                rel_graph = g[stype, etype, dtype]
                if stype not in src_inputs or dtype not in dst_inputs:
                    continue
                dstdata = self.mods[etype](
                    rel_graph,
                    (src_inputs[stype], dst_inputs[dtype]),
                    edge_attr[etype],
                    *mod_args.get(etype, ()),
                    **mod_kwargs.get(etype, {}))
                outputs[dtype].append(dstdata)
        else:
            for stype, etype, dtype in g.canonical_etypes:
                rel_graph = g[stype, etype, dtype]
                if stype not in inputs:
                    continue
                dstdata = self.mods[etype](
                    rel_graph,
                    (inputs[stype], inputs[dtype]),
                    edge_attr[etype],
                    *mod_args.get(etype, ()),
                    **mod_kwargs.get(etype, {}))
                outputs[dtype].append(dstdata)
        rsts = {}
        for nty, alist in outputs.items():
            if len(alist) != 0:
                if nty == 'user':
                    rsts[nty] = self.semantic_attention1(alist)
                else:
                    rsts[nty] = self.semantic_attention2(alist)
        return rsts


class HMGNN(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats, rel_names, num_heads, num_users=5000, use_transformer=True):
        super().__init__()
        self.conv1 = HeteroGraph({
            rel: Edge_level(in_feats, hid_feats, num_heads=num_heads, )
            for rel in rel_names}, in_size_sem=hid_feats, num_head=num_heads)
        self.conv2 = HeteroGraph({
            rel: Edge_level(hid_feats * num_heads, out_feats, num_heads=num_heads, )
            for rel in rel_names}, in_size_sem=out_feats, num_head=num_heads)
        
        self.use_transformer = use_transformer
        
        if self.use_transformer:
            # Sparse Transformer Encoder
            self.transformer_dim = out_feats * num_heads 
            self.sparse_transformer = SparseTransformerBlock(
                d_model=self.transformer_dim, 
                num_heads=num_heads, 
                window_size=16,  # 保持合理的窗口大小
                dilation=3,      # 保持合理的空洞间隔
                dropout=0.2      # 增加 dropout 防止过拟合
            )
            
            # 使用简单的可学习位置嵌入，而非拉普拉斯 PE
            # 这对于 transductive setting 更有效
            self.pos_embed_user = nn.Embedding(num_users, self.transformer_dim)
            nn.init.normal_(self.pos_embed_user.weight, std=0.02)
            
            # 门控融合机制 - 让模型自己决定何时使用 Transformer
            self.gate_network = nn.Sequential(
                nn.Linear(self.transformer_dim * 2, self.transformer_dim),
                nn.ReLU(),
                nn.Linear(self.transformer_dim, 1),
                nn.Sigmoid()
            )

        self.lin = nn.Linear(out_feats * num_heads, out_feats)
        self.lin2 = nn.Linear(out_feats, out_feats)
        self.relu = nn.ReLU()

    def forward(self, graph, inputs, edge_attr):
        h = self.conv1(graph, inputs, edge_attr)
        h = {k: v.reshape(v.shape[0], -1) for k, v in h.items()}
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h, edge_attr)
        h = {k: v.reshape(v.shape[0], -1) for k, v in h.items()}
        h = {k: F.relu(v) for k, v in h.items()}
        
        if self.use_transformer:
            h_fused = {}
            
            # 只对 user 节点应用 Transformer
            if 'user' in h:
                h_gnn = h['user']  # [num_users, dim]
                num_users = h_gnn.shape[0]
                
                # 添加可学习的位置嵌入
                user_ids = torch.arange(num_users, device=h_gnn.device)
                pos_embed = self.pos_embed_user(user_ids)
                feat_with_pe = h_gnn + pos_embed
                
                # 通过 Transformer
                feat_with_pe = feat_with_pe.unsqueeze(0)  # [1, num_users, dim]
                h_transformer = self.sparse_transformer(feat_with_pe)
                h_transformer = h_transformer.squeeze(0)  # [num_users, dim]
                
                # 门控融合: gate 决定每个节点使用多少 Transformer 信息
                # 拼接 GNN 和 Transformer 特征
                combined = torch.cat([h_gnn, h_transformer], dim=-1)  # [num_users, 2*dim]
                gate = self.gate_network(combined)  # [num_users, 1]
                
                # 门控融合: h = gate * h_transformer + (1 - gate) * h_gnn
                # 对于大多数节点，gate 会接近 0（主要用 GNN）
                # 只有需要全局信息的节点，gate 才会变大
                h_fused['user'] = gate * h_transformer + (1 - gate) * h_gnn
            
            # poi 节点直接使用 GNN 输出
            if 'poi' in h:
                h_fused['poi'] = h['poi']
        else:
            h_fused = h
        
        # 最终的线性投影
        h_fused = {k: self.lin(v) for k, v in h_fused.items()}
        h_fused = {k: self.lin2(v) for k, v in h_fused.items()}
        return h_fused


class Model(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, rel_names, num_heads, num_users=5000, use_transformer=True):
        super().__init__()
        self.rel_names = rel_names
        self.sage = HMGNN(in_features, hidden_features, out_features, rel_names, num_heads, num_users=num_users, use_transformer=use_transformer)
        self.pred = HeteroDotProductPredictor()
        self.lin = nn.Linear(out_features, out_features)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.fc_list_node = nn.ModuleList([nn.Linear(feats_dim, in_features, bias=True)
                                           for feats_dim in [128, 128]])
        self.fc_list_edg = nn.ModuleList([nn.Linear(feats_dim, 5, bias=True) for feats_dim in   [5, 4, 2, 1, 1, 2, 4]])
        for fc in self.fc_list_node:
            nn.init.xavier_normal_(fc.weight,gain=1.414)
        for fc in self.fc_list_edg:
            nn.init.xavier_normal_(fc.weight,gain=1.414)

    def predict(self, h, pos_edge_index, neg_edge_index):
        edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=-1)
        logits = cosine_similarity(h[edge_index[0]], h[edge_index[1]])
        logits_2 = self.relu(logits)
        return logits_2

    def forward(self, g, neg_g, node_feat, edge_attr, etype):
        feat2 = {}
        feat2['user'] = self.relu(self.fc_list_node[0](node_feat['user']))
        feat2['poi'] = self.relu(self.fc_list_node[1](node_feat['poi']))
        i = 0
        edge_attr_new = {}
        for edg in self.rel_names:
            edge_attr_new[edg] = self.relu(self.fc_list_edg[i](edge_attr[edg]))
            i += 1
        h = self.sage(g, feat2, edge_attr_new)
        return self.pred(g, h, etype), self.pred(neg_g, h, etype), h, contrastive_loss(h['user'], g)

