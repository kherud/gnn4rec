import random
from typing import Optional

import torch
import torch.nn as nn
from torch_scatter import scatter_mean
from torch_geometric.data import NeighborSampler, Data


class Dataloader:
    def __init__(self,
                 edges,
                 features,
                 tokenizer,
                 k_hops: int = 4,
                 n_min_nodes: int = 10000,
                 max_text_len: Optional[int] = None):

        self.edges = edges
        self.authors = list(edges.keys())

        # we need this for pytorch geometric
        edge_from, edge_to = [], []
        for node in edges:
            for neighbor in edges[node]:
                edge_from.append(node)
                edge_to.append(neighbor)
        edges = torch.tensor([edge_from, edge_to], dtype=torch.long)
        self.graph = Data(edge_index=edges)
        self.neighbor_sampler = NeighborSampler(self.graph.edge_index, sizes=[-1, -1])

        self.features = features
        self.tokenizer = tokenizer

        self.max_text_len = max_text_len
        self.k_hops = k_hops
        self.n_min_nodes = n_min_nodes

    def get(self, author, text):
        text = self.tokenizer.encode(text).ids
        # truncation
        if self.max_text_len is not None:
            text = text[:self.max_text_len]
        # padding
        if len(text) < 5:
            text += (5 - len(text)) * [1]

        # find k-hop neighborhood
        nodes = {author: 0}
        seen = set()
        for hop in range(self.k_hops):
            for node in nodes.copy():
                if node in seen:
                    continue
                for neighbor in self.edges[node]:
                    nodes.setdefault(neighbor, len(nodes))
                seen.add(node)

        # if there are not enough neighbors, add random candidates
        while len(nodes) < self.n_min_nodes <= len(self.authors):
            node = random.choice(self.authors)
            nodes.setdefault(node, len(nodes))

        node_ids = list(sorted(nodes, key=nodes.get))

        _, node_ids, adjacencies = self.neighbor_sampler.sample(node_ids)

        # build features
        keywords, keyword_mask = [], []
        for i, node_id in enumerate(node_ids):
            node_id = node_id.item()
            keywords.extend(self.features[node_id])
            keyword_mask.extend([i] * len(self.features[node_id]))

        text = torch.tensor(text, dtype=torch.long).unsqueeze(0)
        keywords = torch.tensor(keywords, dtype=torch.long)
        keyword_mask = torch.tensor(keyword_mask, dtype=torch.long)

        return node_ids, adjacencies, text, keywords, keyword_mask


class GNN(torch.nn.Module):
    def __init__(self,
                 n_tokens: int,
                 n_keywords: int,
                 n_authors: int,
                 n_layers: int = 2,
                 hidden_size: int = 256,
                 p_dropout: float = 0.0):
        super().__init__()

        self.n_layers = n_layers
        self.n_keywords = n_keywords
        self.n_authors = n_authors

        self.activation = nn.GELU()

        self.author_embedding = nn.Embedding(n_authors, hidden_size // 2)
        self.keyword_embedding = nn.Embedding(n_keywords, hidden_size // 2)

        self.author_norm = nn.LayerNorm(hidden_size // 2)
        self.keyword_norm = nn.LayerNorm(hidden_size // 2)

        self.convs = torch.nn.ModuleList()
        for i in range(n_layers):
            self.convs.append(GraphConvolution(hidden_size=hidden_size,
                                               p_dropout=p_dropout))

        self.text_encoder = CNN(n_tokens, hidden_size, hidden_size, p_dropout=p_dropout)

        self.q1 = nn.Linear(hidden_size, hidden_size // 2)
        self.k1 = nn.Linear(hidden_size // 2, hidden_size // 2)
        self.q2 = nn.Linear(hidden_size // 2, hidden_size // 2)
        self.k2 = nn.Linear(hidden_size // 2, hidden_size // 2)

        self.classifier = Linear(input_size=hidden_size, hidden_size=hidden_size, p_dropout=p_dropout)

    def forward(self, n, a, t, k, m):
        p = self.text_encoder(t)

        aq = self.q2(self.activation(self.q1(p))).squeeze(0)
        ak = self.k2(self.activation(self.k1(self.keyword_embedding.weight)))
        # aq = self.q1(p).squeeze(0)
        # ak = self.k1(self.keyword_embedding.weight)
        w = torch.einsum("i,ki->k", aq, ak)

        # print(w[k].shape, self.embedding(k).shape)
        x = self.keyword_embedding(k)
        x = torch.einsum("ij,i->ij", x, w[k])
        xk = scatter_mean(x, m, dim=0)
        # xk = self.keyword_norm(xk)

        xa = self.author_embedding(n)
        # xa = self.author_norm(xa)

        x = torch.cat((xa, xk), dim=-1)

        for i, (edge_index, _, size) in enumerate(a):
            x = self.convs[i](x, x[:size[1]], edge_index, size[1])

        predictions = self.classifier(p, x).squeeze(1)

        return predictions, w


class GraphConvolution(torch.nn.Module):
    def __init__(self,
                 hidden_size: int,
                 p_dropout: float = 0.1):
        super().__init__()

        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=p_dropout)

        self.linear0 = nn.Linear(hidden_size, hidden_size)
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(2 * hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)

        self.q = nn.Linear(hidden_size, hidden_size)
        self.k = nn.Linear(hidden_size, hidden_size)

        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, source, target, edge_index, dim_size):
        message = self.activation(self.linear0(source))
        message = self.activation(self.linear1(message))
        message = message.index_select(0, edge_index[0])

        # q = self.q(target).index_select(0, edge_index[1])
        # k = self.k(message)
        # a = torch.einsum("ij,ij->i", q, k)
        # a = softmax(a, edge_index[1])
        # message = torch.einsum("i,ij->ij", a, message)

        # aggregated = scatter_sum(message, edge_index[1], dim=0, dim_size=dim_size)
        aggregated = scatter_mean(message, edge_index[1], dim=0, dim_size=dim_size)

        combined = torch.cat((target, aggregated), dim=-1)
        combined = self.activation(self.linear2(combined))
        combined = self.activation(self.linear3(combined))

        # residual connection
        combined += target

        combined = self.norm(combined)
        # combined = self.dropout(combined)

        return combined


class Linear(torch.nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 128, p_dropout: float = 0.1):
        super().__init__()

        self.dropout = nn.Dropout(p=p_dropout)
        self.activation = nn.GELU()

        # self.norm0 = nn.BatchNorm1d(hidden_size)
        # self.norm1 = nn.BatchNorm1d(hidden_size)
        # self.norm2 = nn.BatchNorm1d(hidden_size)
        # self.norm3 = nn.BatchNorm1d(hidden_size)
        # self.norm4 = nn.BatchNorm1d(hidden_size // 2)
        # self.norm0 = nn.LayerNorm(hidden_size)
        # self.norm1 = nn.LayerNorm(hidden_size)
        # self.norm2 = nn.LayerNorm(hidden_size)
        # self.norm3 = nn.LayerNorm(hidden_size)
        # self.norm4 = nn.LayerNorm(hidden_size // 2)

        self.linear0 = nn.Linear(input_size, hidden_size)  # , bias=False
        self.linear1 = nn.Linear(input_size, hidden_size)  # , bias=False
        self.linear2 = nn.Linear(input_size, hidden_size)  # , bias=False
        self.linear3 = nn.Linear(3 * hidden_size, hidden_size)  # , bias=False
        self.linear4 = nn.Linear(hidden_size, hidden_size // 2)  # , bias=False
        self.linear5 = nn.Linear(hidden_size // 2, 1)

    def forward(self, q, x):
        # queries = self.dropout(self.activation(self.linear0(q))).repeat(x1.shape[0], 1)
        # edge_from = self.dropout(self.activation(self.linear1(x1)))
        # edge_to = self.dropout(self.activation(self.linear2(x2)))
        #
        # x = torch.cat((queries, edge_from, edge_to), dim=-1)
        # x = self.dropout(self.activation(self.linear3(x)))
        # x = self.dropout(self.activation(self.linear4(x)))

        queries = self.activation(self.linear0(q)).repeat(x.shape[0] - 1, 1)
        edge_from = self.activation(self.linear1(x[0].unsqueeze(0))).repeat(x.shape[0] - 1, 1)
        edge_to = self.activation(self.linear2(x[1:]))

        x = torch.cat((queries, edge_from, edge_to), dim=-1)
        x = self.activation(self.linear3(x))
        x = self.activation(self.linear4(x))

        return self.linear5(x)


class CNN(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 hidden_size: int,
                 embedding_size: int,
                 p_dropout: float = 0.1):
        super(CNN, self).__init__()

        self.embedding_size = embedding_size
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=self.embedding_size,
                                      padding_idx=1)  # max_norm=5.0

        self.two_gram = TextConvolution(self.embedding_size, 2, 100)
        self.three_gram = TextConvolution(self.embedding_size, 3, 100)
        self.four_gram = TextConvolution(self.embedding_size, 4, 100)

        self.dropout = nn.Dropout(p=p_dropout)
        self.linear = nn.Linear(300, hidden_size, bias=False)
        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, x):
        x = self.embedding(x).transpose(1, 2)

        x2 = self.two_gram(x)
        x3 = self.three_gram(x)
        x4 = self.four_gram(x)

        x = torch.cat((x2, x3, x4), dim=1)

        x = self.linear(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)

        return x


class TextConvolution(nn.Module):
    def __init__(self, input_size: int, filter_size: int, n_filters: int = 100, dropout: float = 0.1):
        super().__init__()

        self.convolution = nn.Conv1d(in_channels=input_size,
                                     out_channels=n_filters,
                                     kernel_size=(filter_size,)) # bias=False # no bias needed since batch norm
        # self.norm = nn.BatchNorm1d(n_filters)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.convolution(x)
        # x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        x, _ = torch.max(x, dim=-1)
        return x
