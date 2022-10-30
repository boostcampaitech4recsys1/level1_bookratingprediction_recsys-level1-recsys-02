import numpy as np
import torch
from ._models import _NGCF


class NGCF(object):
    def __init__(self, args, data):
        self.data_generator = data
        self.adj_mtx = self.data_generator.get_adj_mat()

        self.model = _NGCF

    def train(self):
        """
        Train the model PyTorch style

        Arguments:
        ---------
        model: PyTorch model
        data_generator: Data object
        optimizer: PyTorch optimizer
        """
        self.model.train()
        n_batch = self.data_generator.n_train // self.data_generator.batch_size + 1
        running_loss = 0
        for _ in range(n_batch):
            u, i, j = self.data_generator.sample()
            self.optimizer.zero_grad()
            loss = self.model(u, i, j)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        return running_loss

    def early_stopping(
            self,
            log_value,
            best_value,
            stopping_step,
            flag_step,
            expected_order='asc'
    ):
        """
        Check if early_stopping is needed
        Function copied from original code
        """
        assert expected_order in ['asc', 'des']
        if (expected_order == 'asc' and log_value >= best_value) or (
                expected_order == 'des' and log_value <= best_value):
            stopping_step = 0
            best_value = log_value
        else:
            stopping_step += 1

        if stopping_step >= flag_step:
            print("Early stopping at step: {} log:{}".format(flag_step, log_value))
            should_stop = True
        else:
            should_stop = False

        return best_value, stopping_step, should_stop

    def split_matrix(
            self,
            X,
            n_splits=100
    ):
        """
        Split a matrix/Tensor into n_folds (for the user embeddings and the R matrices)

        Arguments:
        ---------
        X: matrix to be split
        n_folds: number of folds

        Returns:
        -------
        splits: split matrices
        """
        splits = []
        chunk_size = X.shape[0] // n_splits
        for i in range(n_splits):
            start = i * chunk_size
            end = X.shape[0] if i == n_splits - 1 else (i + 1) * chunk_size
            splits.append(X[start:end])

        return splits

    def compute_ndcg_k(
            self,
            pred_items,
            test_items,
            test_indices,
            k
    ):
        """
        Compute NDCG@k

        Arguments:
        ---------
        pred_items: binary tensor with 1s in those locations corresponding to the predicted item interactions
        test_items: binary tensor with 1s in locations corresponding to the real test interactions
        test_indices: tensor with the location of the top-k predicted items
        k: k'th-order

        Returns:
        -------
        NDCG@k
        """
        r = (test_items * pred_items).gather(1, test_indices)
        f = torch.from_numpy(np.log2(np.arange(2, k + 2))).float().to(self.device)

        dcg = (r[:, :k] / f).sum(1)
        dcg_max = (torch.sort(r, dim=1, descending=True)[0][:, :k] / f).sum(1)
        ndcg = dcg / dcg_max

        ndcg[torch.isnan(ndcg)] = 0
        return ndcg

    def eval_model(
            self,
            u_emb,
            i_emb,
            Rtr,
            Rte,
            k
    ):
        """
        Evaluate the model

        Arguments:
        ---------
        u_emb: User embeddings
        i_emb: Item embeddings
        Rtr: Sparse matrix with the training interactions
        Rte: Sparse matrix with the testing interactions
        k : kth-order for metrics

        Returns:
        --------
        result: Dictionary with lists correponding to the metrics at order k for k in Ks
        """

        ue_splits = self.split_matrix(u_emb)
        tr_splits = self.split_matrix(Rtr)
        te_splits = self.split_matrix(Rte)

        recall_k, ndcg_k = [], []

        for ue_f, tr_f, te_f in zip(ue_splits, tr_splits, te_splits):
            scores = torch.mm(ue_f, i_emb.t())

            test_items = torch.from_numpy(te_f.todense()).float().to(self.device)
            non_train_items = torch.from_numpy(1 - (tr_f.todense())).float().to(self.device)
            scores = scores * non_train_items

            _, test_indices = torch.topk(scores, dim=1, k=k)

            pred_items = torch.zeros_like(scores).float()
            pred_items.scatter_(dim=1, index=test_indices, src=torch.ones_like(test_indices).float().to(self.device))

            topk_preds = torch.zeros_like(scores).float()
            topk_preds.scatter_(dim=1, index=test_indices[:, :k], src=torch.ones_like(test_indices).float())

            TP = (test_items * topk_preds).sum(1)
            rec = TP / test_items.sum(1)

            ndcg = self.compute_ndcg_k(pred_items, test_items, test_indices, k)

            recall_k.append(rec)
            ndcg_k.append(ndcg)

        return torch.cat(recall_k).mean(), torch.cat(ndcg_k).mean()
