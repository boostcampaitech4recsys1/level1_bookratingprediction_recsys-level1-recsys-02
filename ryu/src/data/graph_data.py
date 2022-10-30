import os
import random
import pandas as pd
import numpy as np
import scipy.sparse as sp


class GraphDataset(object):
    """
    Dataset for Graph NN model

    Args:
        dataset_paths (list): list of dataset path
        args (argparse): argparse
    """

    def __init__(
            self,
            args, # argparse
    ) -> None:
        super(GraphDataset, self).__init__()
        self.base_path = args.DATA_PATH
        self.batch_size = args.batch_size

        self.n_users, self.n_items = 0, 0
        self.n_train, self.n_test = 0, 0
        self.neg_pools = {}

        self.exist_users = list()

        self.data_info_dict, self.train_graph, self.test_graph = self.load_data(self.base_path)

        self.r_train = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)
        self.r_test = sp.dok_matrix((self.n_users, self.n_items), dtype=np.float32)

        self.train_items, self.test_set = self.get_interation_matrix()

    def load_data(self, base_path):

        def csv2graph(train: pd.DataFrame, test: pd.DataFrame) -> dict:
            train_graph = {k: list(v) for k, v in train.groupby("user_id")["isbn"]}
            test_graph = {k: list(v) for k, v in test.groupby("user_id")["isbn"]}

            return train_graph, test_graph

        train = pd.read_csv(os.path.join(base_path + "train_ratings.csv"))
        test = pd.read_csv(os.path.join(base_path + "test_ratings.csv"))
        sub = pd.read_csv(os.path.join(base_path + "sample_submission.csv"))

        data_info_dict = id_reset(train, test, sub)
        train_graph, test_graph = csv2graph(data_info_dict["train"], data_info_dict["test"])

        for k, v in train_graph:
            items = [int(item) for item in v]
            user_id = k
            self.exist_users.append(user_id)

            self.n_users = max(self.n_users, user_id)
            self.n_items = max(self.n_items, max(items))

            self.n_train += len(items)

        for k, v in test_graph:
            items = [int(item) for item in v]
            self.n_items = max(self.n_items, max(items))
            self.n_test += len(items)

        self.n_users += 1
        self.n_items += 1

        return data_info_dict, train_graph, test_graph

    def get_interation_matrix(self):
        train_items, test_set = {}, {}
        for train_k, train_v in self.train_graph:
            items = [int(item) for item in train_v]
            user_id = train_k

            for item in items:
                self.r_train[user_id, item] = 1.
            train_items[user_id] = items

        for test_k, test_v in self.test_graph:
            items = [int(item) for item in test_v]
            user_id = test_k
            for item in items:
                self.r_test[user_id, item] = 1.0
            test_set[user_id] = items

        return train_items, test_set

    def get_adj_mat(self):
        try:
            adj_mat = sp.load_npz(self.base_path + '/s_adj_mat.npz')
            print('Loaded adjacency-matrix (shape:', adj_mat.shape,') in')

        except Exception:
            print('Creating adjacency-matrix...')
            adj_mat = self.create_adj_mat()
            sp.save_npz(self.base_path + '/s_adj_mat.npz', adj_mat)
        return adj_mat


    def create_adj_mat(self):
        adj_mat = sp.dok_matrix((
            self.n_users + self.n_items, self.n_users + self.n_items),
            dtype=np.float32
        )
        adj_mat = adj_mat.tolil()
        r = self.r_train.tolil()  # to list of lists

        adj_mat[:self.n_users, self.n_users:] = r
        adj_mat[self.n_users:, :self.n_users] = r.T
        adj_mat = adj_mat.todok()

        def normalized_adj_single(adj):
            rowsum = np.array(adj.sum(1))
            d_inv = np.power(rowsum, -.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            d_mat_inv = sp.diags(d_inv)
            norm_adj = d_mat_inv.dot(adj).dot(d_mat_inv)

            return norm_adj.tocoo()

        ngcf_adj_mat = normalized_adj_single(adj_mat)
        return ngcf_adj_mat.tocsr()


    def negative_pool(self):
        for user_id in self.train_items.keys():
            neg_items = list(set(range(self.n_items)) - set(self.train_items[user_id]))
            pools = [random.choice(neg_items) for _ in range(100)]
            self.neg_pools[user_id] = pools

    def sample(self):
        if self.batch_size <= self.n_users:
            users = random.sample(self.exist_users, self.batch_size)
        else:
            users = [random.choice(self.exist_users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(user_id, num):
            pos_items = self.train_items[user_id]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_user_id(user_id, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
                if neg_id not in self.train_items[user_id] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items

        def sample_neg_items_for_u_from_pools(user_id, num):
            neg_items = list(set(self.neg_pools[user_id]) - set(self.train_items[user_id]))
            return random.sample(neg_items, num)

        pos_items, neg_items = [], []
        for user_id in users:
            pos_items += sample_neg_items_for_user_id(user_id, 1)
            neg_items += sample_neg_items_for_user_id(user_id, 1)

        return users, pos_items, neg_items

    def get_num_users_items(self):
        return self.n_users, self.n_items


def id_reset(train: pd.DataFrame, test: pd.DataFrame, sub:pd.DataFrame) -> dict:
    train_idx2user = {idx: user_id for idx, user_id in enumerate(train["user_id"].unique())}
    train_idx2isbn = {idx: isbn for idx, isbn in enumerate(train["isbn"].unique())}
    train_user2idx = {user_id: idx for idx, user_id in train_idx2user.items()}
    train_isbn2idx = {isbn: idx for idx, isbn in train_idx2isbn.items()}

    test_idx2user = {idx: user_id for idx, user_id in enumerate(test["user_id"].unique())}
    test_idx2isbn = {idx: isbn for idx, isbn in enumerate(test["isbn"].unique())}
    test_user2idx = {user_id: idx for idx, user_id in test_idx2user.items()}
    test_isbn2idx = {isbn: idx for idx, isbn in test_idx2isbn.items()}

    train['user_id'] = train['user_id'].map(train_user2idx)
    train["isbn"] = train["isbn"].map(train_isbn2idx)
    test["user_id"] = test["user_id"].map(test_user2idx)
    test["isbn"] = test["isbn"].map(test_isbn2idx)

    data_info_dict = {
        "train": train,
        "test": test,
        "sub": sub,
        "train_idx2user": train_idx2user,
        "train_idx2isbn": train_idx2isbn,
        "test_idx2user": test_idx2user,
        "test_idx2isbn": test_idx2isbn,
    }

    return data_info_dict
