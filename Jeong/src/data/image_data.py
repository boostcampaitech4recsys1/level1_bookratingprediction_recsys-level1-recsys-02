import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from tqdm import tqdm


class Image_Dataset(Dataset):
    def __init__(self, context_vector, img_vector, label):
        self.context_vector = context_vector
        self.img_vector = img_vector
        self.label = label

    def __len__(self):
        return self.context_vector.shape[0]

    def __getitem__(self, i):
        return {
            "context_vector": torch.tensor(self.context_vector[i], dtype=torch.long),
            "img_vector": torch.tensor(self.img_vector[i], dtype=torch.float32),
            "label": torch.tensor(self.label[i], dtype=torch.float32),
        }


def process_context_data(
    users: pd.DataFrame,
    books: pd.DataFrame,
    rating_train: pd.DataFrame,
    rating_test: pd.DataFrame,
):
    ratings = pd.concat([rating_train, rating_test]).reset_index(drop=True)

    # user에서 활용할 데이터 목록
    user_cols = ["user_id", "age", "location_state", "location_country"]

    # book에서 활용할 데이터 목록
    book_cols = [
        "isbn",
        "book_author",
        "year_of_publication",
        "publisher",
        "category_high",
        "isbn_country",
    ]

    # 활용할 context 목록
    context_cols = user_cols.copy()
    context_cols.extend(book_cols)
    context_cols.remove("user_id")
    context_cols.remove("isbn")

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


def image_vector(path):
    img = Image.open(path)
    scale = transforms.Resize((32, 32))
    tensor = transforms.ToTensor()
    img_fe = Variable(tensor(scale(img)))
    return img_fe


def process_img_data(df, books, user2idx, isbn2idx, train=False):
    books_ = books.copy()

    if train == True:
        df_ = df.copy()
    else:
        df_ = df.copy()

    df_ = pd.merge(df_, books_[["isbn", "img_path"]], on="isbn", how="left")

    df_["img_path"] = df_["img_path"].apply(lambda x: "data/" + x)
    img_vector_df = df_[["img_path"]].drop_duplicates().reset_index(drop=True).copy()
    data_box = []
    for idx, path in tqdm(enumerate(sorted(img_vector_df["img_path"]))):
        data = image_vector(path)
        if data.size()[0] == 3:
            data_box.append(np.array(data))
        else:
            data_box.append(np.array(data.expand(3, data.size()[1], data.size()[2])))
    img_vector_df["img_vector"] = data_box
    df_ = pd.merge(df_, img_vector_df, on="img_path", how="left")

    df_.drop(["img_path"], axis=1, inplace=True)

    return df_


def image_data_load(args):

    users = pd.read_csv(args.DATA_PATH + "users_processed.csv")
    books = pd.read_csv(args.DATA_PATH + "books_processed.csv")
    train = pd.read_csv(args.DATA_PATH + "train_ratings.csv")
    test = pd.read_csv(args.DATA_PATH + "test_ratings.csv")
    sub = pd.read_csv(args.DATA_PATH + "sample_submission.csv")

    ids = pd.concat([train["user_id"], sub["user_id"]]).unique()
    isbns = pd.concat([train["isbn"], sub["isbn"]]).unique()

    idx2user = {idx: id for idx, id in enumerate(ids)}
    idx2isbn = {idx: isbn for idx, isbn in enumerate(isbns)}

    user2idx = {id: idx for idx, id in idx2user.items()}
    isbn2idx = {isbn: idx for idx, isbn in idx2isbn.items()}

    # train["user_id"] = train["user_id"].map(user2idx)
    # sub["user_id"] = sub["user_id"].map(user2idx)

    # train["isbn"] = train["isbn"].map(isbn2idx)
    # sub["isbn"] = sub["isbn"].map(isbn2idx)

    # id idx로 변경 하는 부분 추가
    train["user_id"] = train["user_id"].map(user2idx)
    sub["user_id"] = sub["user_id"].map(user2idx)
    test["user_id"] = test["user_id"].map(user2idx)
    users["user_id"] = users["user_id"].map(user2idx)

    train["isbn"] = train["isbn"].map(isbn2idx)
    sub["isbn"] = sub["isbn"].map(isbn2idx)
    test["isbn"] = test["isbn"].map(isbn2idx)
    books["isbn"] = books["isbn"].map(isbn2idx)

    idx, train, test = process_context_data(users, books, train, test)
    img_train = process_img_data(train, books, user2idx, isbn2idx, train=True)
    img_test = process_img_data(test, books, user2idx, isbn2idx, train=False)

    field_dims_list = [len(user2idx), len(isbn2idx)]
    for k, v in idx.items():
        field_dims_list.append(len(v))

    field_dims = np.array(field_dims_list, dtype=np.uint32)
    data = {
        "train": train,
        "test": test.drop(["rating"], axis=1),
        "field_dims": field_dims,
        "users": users,
        "books": books,
        "sub": sub,
        "idx2user": idx2user,
        "idx2isbn": idx2isbn,
        "user2idx": user2idx,
        "isbn2idx": isbn2idx,
        "img_train": img_train,
        "img_test": img_test,
    }

    return data


def image_data_split(args, data):
    X_train, X_valid, y_train, y_valid = train_test_split(
        data["img_train"].drop(["rating"], axis=1),
        data["img_train"]["rating"],
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


def image_data_loader(args, data):
    train_dataset = Image_Dataset(
        data["X_train"].drop(["img_vector"], axis=1).values,
        data["X_train"]["img_vector"].values,
        data["y_train"].values,
    )
    valid_dataset = Image_Dataset(
        data["X_valid"].drop(["img_vector"], axis=1).values,
        data["X_valid"]["img_vector"].values,
        data["y_valid"].values,
    )
    test_dataset = Image_Dataset(
        data["img_test"].drop(["img_vector", "rating"], axis=1).values,
        data["img_test"]["img_vector"].values,
        data["img_test"]["rating"].values,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.BATCH_SIZE, num_workers=0, shuffle=True
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.BATCH_SIZE, num_workers=0, shuffle=True
    )
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.BATCH_SIZE, num_workers=0, shuffle=False
    )
    data["train_dataloader"], data["valid_dataloader"], data["test_dataloader"] = (
        train_dataloader,
        valid_dataloader,
        test_dataloader,
    )

    return data
