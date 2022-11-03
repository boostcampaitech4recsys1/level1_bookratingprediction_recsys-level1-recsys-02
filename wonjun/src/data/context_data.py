import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset

def age_map(x: int) -> int:
    x = int(x)
    if x < 20:
        return 1
    elif x >= 20 and x < 30:
        return 2
    elif x >= 30 and x < 40:
        return 3
    elif x >= 40 and x < 50:
        return 4
    elif x >= 50 and x < 60:
        return 5
    else:
        return 6

def calc_mean_rating(train: pd.DataFrame):
    train_mean_rating = train["rating"].mode()[0]

    # user2rating = train.groupby('user_id')['rating'].mode()[0]
    user2rating = train.groupby('user_id')['rating'].apply(lambda x:x.value_counts().index[0])

    user2rating = user2rating.to_dict()

    #isbn2rating = train.groupby('isbn')['rating'].mode()[0]
    isbn2rating = train.groupby('isbn')['rating'].apply(lambda x:x.value_counts().index[0])
    isbn2rating = isbn2rating.to_dict()

    return train_mean_rating, user2rating, isbn2rating


def process_context_data(users, books, ratings1, ratings2):
    #users['location_city'] = users['location'].apply(lambda x: x.split(',')[0])
    #users['location_state'] = users['location'].apply(lambda x: x.split(',')[1])
    #users['location_country'] = users['location'].apply(lambda x: x.split(',')[2])
    #users = users.drop(['location'], axis=1)

    ratings = pd.concat([ratings1, ratings2]).reset_index(drop=True)

    # 인덱싱 처리된 데이터 조인
    context_df = ratings.merge(users, on='user_id', how='left').merge(books[['isbn', 'category_high', 'publisher', 'isbn_country', 'book_author', 'year_of_publication']], on='isbn', how='left')
    train_df = ratings1.merge(users, on='user_id', how='left').merge(books[['isbn', 'category_high', 'publisher', 'isbn_country', 'book_author', 'year_of_publication']], on='isbn', how='left')
    test_df = ratings2.merge(users, on='user_id', how='left').merge(books[['isbn', 'category_high', 'publisher', 'isbn_country', 'book_author', 'year_of_publication']], on='isbn', how='left')

    # 인덱싱 처리
    #loc_city2idx = {v:k for k,v in enumerate(context_df['location_city'].unique())}
    #loc_state2idx = {v:k for k,v in enumerate(context_df['location_state'].unique())}
    loc_country2idx = {v:k for k,v in enumerate(context_df['location_country'].unique())}

    #train_df['location_city'] = train_df['location_city'].map(loc_city2idx)
    #train_df['location_state'] = train_df['location_state'].map(loc_state2idx)
    train_df['location_country'] = train_df['location_country'].map(loc_country2idx)
    #test_df['location_city'] = test_df['location_city'].map(loc_city2idx)
    #test_df['location_state'] = test_df['location_state'].map(loc_state2idx)
    test_df['location_country'] = test_df['location_country'].map(loc_country2idx)

    #train_df['age'] = train_df['age'].dropna(axis=0)
    #train_df['age'] = train_df['age'].apply(age_map)
    #test_df['age'] = test_df['age'].dropna(axis=0)
    #test_df['age'] = test_df['age'].apply(age_map)

    # book 파트 인덱싱
    category2idx = {v:k for k,v in enumerate(context_df['category_high'].unique())}
    publisher2idx = {v:k for k,v in enumerate(context_df['publisher'].unique())}
    language2idx = {v:k for k,v in enumerate(context_df['isbn_country'].unique())}
    author2idx = {v:k for k,v in enumerate(context_df['book_author'].unique())}
    publication2idx = {v:k for k,v in enumerate(context_df['year_of_publication'].unique())}

    train_df['category_high'] = train_df['category_high'].map(category2idx)
    train_df['publisher'] = train_df['publisher'].map(publisher2idx)
    train_df['isbn_country'] = train_df['isbn_country'].map(language2idx)
    train_df['book_author'] = train_df['book_author'].map(author2idx)
    train_df["year_of_publication"] = train_df["year_of_publication"].map(publication2idx)
    test_df['category_high'] = test_df['category_high'].map(category2idx)
    test_df['publisher'] = test_df['publisher'].map(publisher2idx)
    test_df['isbn_country'] = test_df['isbn_country'].map(language2idx)
    test_df['book_author'] = test_df['book_author'].map(author2idx)
    test_df["year_of_publication"] = test_df["year_of_publication"].map(publication2idx)

    idx = {
        #"loc_city2idx":loc_city2idx,
        #"loc_state2idx":loc_state2idx,
        "loc_country2idx": loc_country2idx,
        "category2idx": category2idx,
        "publisher2idx": publisher2idx,
        "language2idx": language2idx,
        "author2idx": author2idx,
        "publication2idx": publication2idx,
    }

    return idx, train_df, test_df


def context_data_load(args):

    ######################## DATA LOAD
    users = pd.read_csv(args.DATA_PATH + 'userspp2.csv')
    books = pd.read_csv(args.DATA_PATH + 'bookspp.csv')
    train = pd.read_csv(args.DATA_PATH + 'train_ppp.csv')
    test = pd.read_csv(args.DATA_PATH + 'test_ratings.csv')
    sub = pd.read_csv(args.DATA_PATH + 'sample_submission.csv')

    # unique_user_ids = train["user_id"].unique()
    # unique_isbns = train["isbn"].unique()

    ids = pd.concat([train['user_id'], sub['user_id']]).unique()
    isbns = pd.concat([train['isbn'], sub['isbn']]).unique()

    idx2user = {idx:id for idx, id in enumerate(ids)}
    idx2isbn = {idx:isbn for idx, isbn in enumerate(isbns)}

    user2idx = {id:idx for idx, id in idx2user.items()}
    isbn2idx = {isbn:idx for idx, isbn in idx2isbn.items()}

    # train_mean_rating, user2rating, isbn2rating = calc_mean_rating(train)

    train['user_id'] = train['user_id'].map(user2idx)
    sub['user_id'] = sub['user_id'].map(user2idx)
    test['user_id'] = test['user_id'].map(user2idx)
    users['user_id'] = users['user_id'].map(user2idx)

    train['isbn'] = train['isbn'].map(isbn2idx)
    sub['isbn'] = sub['isbn'].map(isbn2idx)
    test['isbn'] = test['isbn'].map(isbn2idx)
    books['isbn'] = books['isbn'].map(isbn2idx)

    idx, context_train, context_test = process_context_data(users, books, train, test)
    field_dims = np.array([len(user2idx), len(isbn2idx),
                            len(idx['loc_country2idx']),
                            len(idx['category2idx']), len(idx['publisher2idx']), len(idx['language2idx']), len(idx['author2idx']), len(idx["publication2idx"])], dtype=np.uint32)

    data = {
            'train':context_train,
            'test':context_test.drop(['rating'], axis=1),
            'field_dims':field_dims,
            'users':users,
            'books':books,
            'sub':sub,
            'idx2user':idx2user,
            'idx2isbn':idx2isbn,
            'user2idx':user2idx,
            'isbn2idx':isbn2idx,
            #"isbn2rating": isbn2rating,
            #"user2rating": user2rating,
            #"train_mean_rating": train_mean_rating,
            #'unique_user_ids': unique_user_ids,
            #'unique_isbns': unique_isbns
            }


    return data


def context_data_split(args, data):
    X_train, X_valid, y_train, y_valid = train_test_split(
                                                        data['train'].drop(['rating'], axis=1),
                                                        data['train']['rating'],
                                                        test_size=args.TEST_SIZE,
                                                        random_state=args.SEED,
                                                        shuffle=True
                                                        )
    # X = data["train"].drop(["rating"], axis=1)
    # y = data["train"]["rating"]
    
    # X_train = X.iloc[:int(0.75 * len(X))]
    # X_valid = X.iloc[int(0.75 * len(X)):]
    
    # y_train = y.iloc[:int(0.75 * len(X))]
    # y_valid = y.iloc[int(0.75 * len(X)):]

    data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
    return data

def context_data_loader(args, data):
    train_dataset = TensorDataset(torch.LongTensor(data['X_train'].values), torch.LongTensor(data['y_train'].values))
    valid_dataset = TensorDataset(torch.LongTensor(data['X_valid'].values), torch.LongTensor(data['y_valid'].values))
    test_dataset = TensorDataset(torch.LongTensor(data['test'].values))

    train_dataloader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=args.DATA_SHUFFLE)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.BATCH_SIZE, shuffle=args.DATA_SHUFFLE)
    test_dataloader = DataLoader(test_dataset, batch_size=args.BATCH_SIZE, shuffle=False)

    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader

    return data
