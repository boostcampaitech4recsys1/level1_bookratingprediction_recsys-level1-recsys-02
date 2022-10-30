import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F


def rmse(real: list, predict: list) -> float:
    pred = np.array(predict)
    return np.sqrt(np.mean((real-pred) ** 2))


class RMSELoss(torch.nn.Module):
    def __init__(self):
        super(RMSELoss,self).__init__()
        self.eps = 1e-6

    def forward(self, x, y):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(x, y)+self.eps)
        return loss


class FactorizationMachine(nn.Module):

    def __init__(self, reduce_sum:bool=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x: torch.Tensor):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix


class FactorizationMachine_v(nn.Module):

    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.v = nn.Parameter(torch.rand(input_dim, latent_dim), requires_grad = True)
        self.linear = nn.Linear(input_dim, 1, bias=True)

    def forward(self, x):
        linear = self.linear(x)
        square_of_sum = torch.mm(x, self.v) ** 2
        sum_of_square = torch.mm(x ** 2, self.v ** 2)
        pair_interactions = torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)
        output = linear + (0.5 * pair_interactions)
        return output


class FeaturesEmbedding(nn.Module):

    def __init__(self, field_dims: np.ndarray, embed_dim: int):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x: torch.Tensor):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)


class FeaturesLinear(nn.Module):

    def __init__(self, field_dims: np.ndarray, output_dim: int=1):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

    def forward(self, x: torch.Tensor):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias


class _FactorizationMachineModel(nn.Module):

    def __init__(self, field_dims: np.ndarray, embed_dim: int):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.linear = FeaturesLinear(field_dims)
        self.fm = FactorizationMachine(reduce_sum=True)

    def forward(self, x: torch.Tensor):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = self.linear(x) + self.fm(self.embedding(x))
        # return torch.sigmoid(x.squeeze(1))
        return x.squeeze(1)


class FieldAwareFactorizationMachine(nn.Module):

    def __init__(self, field_dims: np.ndarray, embed_dim: int):
        super().__init__()
        self.num_fields = len(field_dims)
        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(sum(field_dims), embed_dim) for _ in range(self.num_fields)
        ])
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        for embedding in self.embeddings:
            torch.nn.init.xavier_uniform_(embedding.weight.data)

    def forward(self, x: torch.Tensor):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets, dtype= np.long).unsqueeze(0)
        xs = [self.embeddings[i](x) for i in range(self.num_fields)]
        ix = list()
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                ix.append(xs[j][:, i] * xs[i][:, j])
        ix = torch.stack(ix, dim=1)
        return ix


class _FieldAwareFactorizationMachineModel(nn.Module):

    def __init__(self, field_dims: np.ndarray, embed_dim: int):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)
        self.ffm = FieldAwareFactorizationMachine(field_dims, embed_dim)

    def forward(self, x: torch.Tensor):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        ffm_term = torch.sum(torch.sum(self.ffm(x), dim=1), dim=1, keepdim=True)
        x = self.linear(x) + ffm_term
        # return torch.sigmoid(x.squeeze(1))
        return x.squeeze(1)


class MultiLayerPerceptron(nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)


class _NeuralCollaborativeFiltering(nn.Module):

    def __init__(self, field_dims, user_field_idx, item_field_idx, embed_dim, mlp_dims, dropout):
        super().__init__()
        self.user_field_idx = user_field_idx
        self.item_field_idx = item_field_idx
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout, output_layer=False)
        self.fc = torch.nn.Linear(mlp_dims[-1] + embed_dim, 1)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_user_fields)``
        """
        x = self.embedding(x)
        user_x = x[:, self.user_field_idx].squeeze(1)
        item_x = x[:, self.item_field_idx].squeeze(1)
        gmf = user_x * item_x
        x = self.mlp(x.view(-1, self.embed_output_dim))
        x = torch.cat([gmf, x], dim=1)
        x = self.fc(x).squeeze(1)

        return x


class _WideAndDeepModel(nn.Module):

    def __init__(self, field_dims: np.ndarray, embed_dim: int, mlp_dims: tuple, dropout: float):
        super().__init__()
        self.linear = FeaturesLinear(field_dims)
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)

    def forward(self, x: torch.Tensor):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        x = self.linear(x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        return x.squeeze(1)


class CrossNetwork(nn.Module):

    def __init__(self, input_dim: int, num_layers: int):
        super().__init__()
        self.num_layers = num_layers
        self.w = torch.nn.ModuleList([
            torch.nn.Linear(input_dim, 1, bias=False) for _ in range(num_layers)
        ])
        self.b = torch.nn.ParameterList([
            torch.nn.Parameter(torch.zeros((input_dim,))) for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        x0 = x
        for i in range(self.num_layers):
            xw = self.w[i](x)
            x = x0 * xw + self.b[i] + x
        return x


class _DeepCrossNetworkModel(nn.Module):
    """
    A pytorch implementation of Deep & Cross Network.
    Reference:
        R Wang, et al. Deep & Cross Network for Ad Click Predictions, 2017.
    """

    def __init__(self, field_dims: np.ndarray, embed_dim: int, num_layers: int, mlp_dims: tuple, dropout: float):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.cn = CrossNetwork(self.embed_output_dim, num_layers)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout, output_layer=False)
        self.cd_linear = nn.Linear(mlp_dims[0], 1, bias=False)

    def forward(self, x: torch.Tensor):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x).view(-1, self.embed_output_dim)
        x_l1 = self.cn(embed_x)
        x_out = self.mlp(x_l1)
        p = self.cd_linear(x_out)
        return p.squeeze(1)


class _NGCF(nn.Module):
    """
    A pytorch implementation of NGCF
    """

    def __init__(
            self,
            n_users,
            n_items,
            emb_dim,
            layers,
            reg,
            node_dropout,
            mess_dropout,
            adj_mtx,
            device,
    ):
        super().__init__()

        self.n_users = n_users
        self.n_items = n_items
        self.emb_dim = emb_dim
        self.l_matrix = adj_mtx
        self.l_plus_i_matrix = adj_mtx + sp.eye(adj_mtx.shape[0])
        self.reg = reg
        self.layers = layers
        self.n_layers = len(self.layers)
        self.node_dropout = node_dropout
        self.mess_dropout = mess_dropout
        self.device = device

        self.weight_dict= self._init_weights()

        self.L_plus_l = self._convert_sp_mat_to_sp_tensor(self.l_plus_i_matrix)
        self.L = self._convert_sp_mat_to_sp_tensor(self.l_matrix)

    def _init_weights(self):
        print("Initializing weights...")
        weight_dict = nn.ParameterDict()

        initializer = torch.nn.init.xavier_uniform_

        weight_dict['user_embedding'] = nn.Parameter(initializer(torch.empty(self.n_users, self.emb_dim).to(self.device)))
        weight_dict['item_embedding'] = nn.Parameter(initializer(torch.empty(self.n_items, self.emb_dim).to(self.device)))

        weight_size_list = [self.emb_dim] + self.layers

        for k in range(self.n_layers):
            weight_dict['W_one_%d' % k] = nn.Parameter(
                initializer(torch.empty(weight_size_list[k], weight_size_list[k + 1]).to(self.device)))
            weight_dict['b_one_%d' % k] = nn.Parameter(initializer(torch.empty(1, weight_size_list[k + 1]).to(self.device)))

            weight_dict['W_two_%d' % k] = nn.Parameter(
                initializer(torch.empty(weight_size_list[k], weight_size_list[k + 1]).to(self.device)))
            weight_dict['b_two_%d' % k] = nn.Parameter(initializer(torch.empty(1, weight_size_list[k + 1]).to(self.device)))

        return weight_dict

    def _convert_sp_mat_to_sp_tensor(self, X):
        """
        Convert scipy sparse matrix to PyTorch sparse matrix

        Arguments:
        ----------
        X = Adjacency matrix, scipy sparse matrix
        """
        coo = X.tocoo().astype(np.float32)
        i = torch.LongTensor(np.mat([coo.row, coo.col]))
        v = torch.FloatTensor(coo.data)
        res = torch.sparse.FloatTensor(i, v, coo.shape).to(self.device)
        return res

    def _droupout_sparse(self, X):
        """
        Drop individual locations in X

        Arguments:
        ---------
        X = adjacency matrix (PyTorch sparse tensor)
        dropout = fraction of nodes to drop
        noise_shape = number of non non-zero entries of X
        """
        node_dropout_mask = ((self.node_dropout) + torch.rand(X._nnz())).floor().bool().to(self.device)
        i = X.coalesce().indices()
        v = X.coalesce()._values()
        i[:, node_dropout_mask] = 0
        v[node_dropout_mask] = 0
        X_dropout = torch.sparse.FloatTensor(i, v, X.shape).to(X.device)

        return X_dropout.mul(1 / (1 - self.node_dropout))

    def forward(self, u, i, j):
        """
        Computes the forward pass

        Arguments:
        ---------
        u = user
        i = positive item (user interacted with item)
        j = negative item (user did not interact with item)
        """
        L_plus_I_hat = self._droupout_sparse(self.L_plus_I) if self.node_dropout > 0 else self.L_plus_I
        L_hat = self._droupout_sparse(self.L) if self.node_dropout > 0 else self.L

        ego_embeddings = torch.cat([self.weight_dict['user_embedding'], self.weight_dict['item_embedding']], 0)

        final_embeddings = [ego_embeddings]

        for k in range(self.n_layers):
            side_L_plus_I_embeddings = torch.sparse.mm(L_plus_I_hat, ego_embeddings)

            simple_embeddings = torch.matmul(
                side_L_plus_I_embeddings,
                self.weight_dict['W_one_%d' % k]
            ) + self.weight_dict['b_one_%d' % k]

            side_L_embeddings = torch.sparse.mm(L_hat, ego_embeddings)

            interaction_embeddings = torch.mul(ego_embeddings, side_L_embeddings)

            interaction_embeddings = torch.matmul(
                interaction_embeddings,
                self.weight_dict['W_two_%d' % k]
            ) + self.weight_dict['b_two_%d' % k]

            ego_embeddings = F.leaky_relu(simple_embeddings + interaction_embeddings)

            mess_dropout_mask = nn.Dropout(self.mess_dropout)
            ego_embeddings = mess_dropout_mask(ego_embeddings)

            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)

            final_embeddings.append(norm_embeddings)

        final_embeddings = torch.cat(final_embeddings, 1)

        u_final_embeddings, i_final_embeddings = final_embeddings.split([self.n_users, self.n_items], 0)

        u_final_embeddings = nn.Parameter(u_final_embeddings)
        i_final_embeddings = nn.Parameter(i_final_embeddings)

        u_emb = u_final_embeddings[u]
        p_emb = i_final_embeddings[i]
        n_emb = i_final_embeddings[j]

        y_ui = torch.mul(u_emb, p_emb).sum(dim=1)
        y_uj = torch.mul(u_emb, n_emb).sum(dim=1)

        log_prob = (torch.log(torch.sigmoid(y_ui - y_uj))).mean()
        bpr_loss = -log_prob

        if self.reg > 0.:
            l2norm = (torch.sum(u_emb ** 2) / 2. + torch.sum(p_emb ** 2) / 2. + torch.sum(n_emb ** 2) / 2.) / \
                     u_emb.shape[0]
            l2reg = self.reg * l2norm
            bpr_loss += l2reg

        return bpr_loss
