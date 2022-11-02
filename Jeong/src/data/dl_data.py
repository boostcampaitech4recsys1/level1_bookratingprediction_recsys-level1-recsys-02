import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset


def process_dl_data(
    users: pd.DataFrame,
    books: pd.DataFrame,
    rating_train: pd.DataFrame,
    rating_test: pd.DataFrame,
):
    ratings = pd.concat([rating_train, rating_test]).reset_index(drop=True)

    # user에서 활용할 데이터 목록
    user_cols = ["user_id", "location_country"]

    # book에서 활용할 데이터 목록
    book_cols = [
        "isbn",
        "book_author",
        "year_of_publication",
        "publisher",
        "category_high",
        "isbn_country",
        "title_ft",
        "summary_ft",
    ]

    # 활용할 context 목록
    context_cols = user_cols.copy()
    context_cols.extend(book_cols)
    context_cols.remove("user_id")
    context_cols.remove("isbn")

    # 텍스트 데이터는 idx화 안함
    context_cols.remove("title_ft")
    context_cols.remove("summary_ft")

    # 인덱싱 처리된 데이터 조인
    context_df = ratings.merge(users[user_cols], on="user_id", how="left").merge(
        books[book_cols],
        on="isbn",
        how="left",
    )
    train_df = rating_train.merge(users[user_cols], on="user_id", how="left").merge(
        books[book_cols],
        on="isbn",
        how="left",
    )
    test_df = rating_test.merge(users[user_cols], on="user_id", how="left").merge(
        books[book_cols],
        on="isbn",
        how="left",
    )

    def col2idx(
        context_df: pd.DataFrame,
        col: str,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ):
        """
        context df의 col에 대해서 idx화 해주기.
        """
        idx_dict = {v: k for k, v in enumerate(context_df[col].unique())}
        train_df[col] = train_df[col].map(idx_dict)
        test_df[col] = test_df[col].map(idx_dict)
        return idx_dict, train_df, test_df

    idx = dict()
    # 인덱싱 처리
    for col in context_cols:
        idx[f"{col}2idx"], train_df, test_df = col2idx(
            context_df, col, train_df, test_df
        )

    return idx, train_df, test_df


def dl_data_load(args):

    ######################## DATA LOAD
    users = pd.read_csv(args.DATA_PATH + "userspp.csv")
    books = pd.read_csv(args.DATA_PATH + "bookspp_text.csv")
    books["title_ft"] = books["title_ft"].apply(
        lambda x: np.fromstring(x, dtype=np.float32, count=100)
    )
    books["summary_ft"] = books["summary_ft"].apply(
        lambda x: np.fromstring(x, dtype=np.float32, count=100)
    )

    train = pd.read_csv(args.DATA_PATH + "train_ppp.csv")
    test = pd.read_csv(args.DATA_PATH + "test_ratings.csv")
    sub = pd.read_csv(args.DATA_PATH + "sample_submission.csv")

    ids = pd.concat([train["user_id"], sub["user_id"]]).unique()
    isbns = pd.concat([train["isbn"], sub["isbn"]]).unique()

    idx2user = {idx: id for idx, id in enumerate(ids)}
    idx2isbn = {idx: isbn for idx, isbn in enumerate(isbns)}

    user2idx = {id: idx for idx, id in idx2user.items()}
    isbn2idx = {isbn: idx for idx, isbn in idx2isbn.items()}

    train["user_id"] = train["user_id"].map(user2idx)
    sub["user_id"] = sub["user_id"].map(user2idx)
    test["user_id"] = test["user_id"].map(user2idx)
    users["user_id"] = users["user_id"].map(user2idx)

    train["isbn"] = train["isbn"].map(isbn2idx)
    sub["isbn"] = sub["isbn"].map(isbn2idx)
    test["isbn"] = test["isbn"].map(isbn2idx)
    books["isbn"] = books["isbn"].map(isbn2idx)

    idx, context_train, context_test = process_dl_data(users, books, train, test)
    field_dims_list = [len(user2idx), len(isbn2idx)]
    for k, v in idx.items():
        field_dims_list.append(len(v))

    # 텍스트 데이터 반영
    # field_dims_list.append(100)
    # field_dims_list.append(100)

    field_dims = np.array(field_dims_list, dtype=np.uint32)

    data = {
        "train": context_train,
        "test": context_test.drop("rating", axis=1),
        "field_dims": field_dims,
        "users": users,
        "books": books,
        "sub": sub,
        "idx2user": idx2user,
        "idx2isbn": idx2isbn,
        "user2idx": user2idx,
        "isbn2idx": isbn2idx,
    }

    return data


def dl_data_split(args, data):
    X_train, X_valid, y_train, y_valid = train_test_split(
        data["train"].drop(["rating"], axis=1),
        data["train"]["rating"],
        test_size=args.TEST_SIZE,
        random_state=args.SEED,
        shuffle=True,
    )
    data["X_train"], data["X_valid"], data["y_train"], data["y_valid"] = (
        X_train,
        X_valid,
        y_train,
        y_valid,
    )
    return data


class Dl_Dataset(Dataset):
    def __init__(
        self, context_vector, title_vector, summary_vector, label=None, is_train=True
    ):
        self.context_vector = context_vector
        self.title_vector = title_vector
        self.summary_vector = summary_vector
        self.train = is_train
        if is_train:
            self.label = label

    def __len__(self):
        return self.context_vector.shape[0]

    def __getitem__(self, i):
        if self.train:
            return {
                "context_vector": torch.tensor(
                    self.context_vector[i], dtype=torch.long
                ),
                "title_vector": torch.tensor(self.title_vector[i], dtype=torch.float32),
                "summary_vector": torch.tensor(
                    self.summary_vector[i], dtype=torch.float32
                ),
                "label": torch.tensor(self.label[i], dtype=torch.float32),
            }
        else:
            return {
                "context_vector": torch.tensor(
                    self.context_vector[i], dtype=torch.long
                ),
                "title_vector": torch.tensor(self.title_vector[i], dtype=torch.float32),
                "summary_vector": torch.tensor(
                    self.summary_vector[i], dtype=torch.float32
                ),
            }


def dl_data_loader(args, data):

    train_dataset = Dl_Dataset(
        data["X_train"].drop(["title_ft", "summary_ft"], axis=1).values,
        data["X_train"]["title_ft"].values,
        data["X_train"]["summary_ft"].values,
        data["y_train"].values,
    )
    valid_dataset = Dl_Dataset(
        data["X_valid"].drop(["title_ft", "summary_ft"], axis=1).values,
        data["X_valid"]["title_ft"].values,
        data["X_valid"]["summary_ft"].values,
        data["y_valid"].values,
    )
    test_dataset = Dl_Dataset(
        data["test"].drop(["title_ft", "summary_ft"], axis=1).values,
        data["test"]["title_ft"].values,
        data["test"]["summary_ft"].values,
        is_train=False,
    )

    # train_dataset = TensorDataset(
    #     torch.LongTensor(data["X_train"].drop(['title_ft', 'summary_ft'], axis=1).values),
    #     torch.LongTensor(data["y_train"].values),
    #     torch.LongTensor(data["X_train"]['title_ft'])
    # )
    # valid_dataset = TensorDataset(
    #     torch.LongTensor(data["X_valid"].values),
    #     torch.LongTensor(data["y_valid"].values),
    # )
    # test_dataset = TensorDataset(torch.LongTensor(data["test"].values))

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.BATCH_SIZE, shuffle=args.DATA_SHUFFLE
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=args.BATCH_SIZE, shuffle=args.DATA_SHUFFLE
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.BATCH_SIZE, shuffle=False
    )

    data["train_dataloader"], data["valid_dataloader"], data["test_dataloader"] = (
        train_dataloader,
        valid_dataloader,
        test_dataloader,
    )

    return data
