import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
from tqdm import tqdm
import re


def process_user_data(users: pd.DataFrame) -> pd.DataFrame:
    country_fix_dict = {
        "usa": {
            "oklahoma",
            "districtofcolumbia",
            "connecticut",
            "worcester",
            "aroostook",
            "texas",
            "kern",
            "orangeco",
            "unitedstatesofamerica",
            "fortbend",
            "alachua",
            "massachusetts",
            "arizona",
            "austin",
            "hawaii",
            "ohio",
            "camden",
            "arkansas",
            "minnesota",
            "losestadosunidosdenorteamerica",
            "us",
            "usanow",
            "northcarolina",
            "maine",
            "colorado",
            "oklahoma",
            "alabama",
            "anystate",
            "districtofcolumbia",
            "unitedstaes",
            "pender",
            "newhampshire",
            "unitedstates",
            "missouri",
            "idaho",
            "ca",
            "newyork",
            "tennessee",
            "stthomasi",
            "dc",
            "washington",
            "illinois",
            "california",
            "michigan",
            "iowa",
            "maryland",
            "newjersey",
            "vanwert",
            "oregon",
        },
        "uk": {
            "alderney",
            "wales",
            "aberdeenshire",
            "bermuda",
            "nottinghamshire",
            "scotland",
            "usacurrentlylivinginengland",
            "england",
            "countycork",
            "alderney",
            "cambridgeshire",
            "middlesex",
            "northyorkshire",
            "westyorkshire",
            "cocarlow",
            "sthelena",
        },
        "japan": {"okinawa"},
        "southkorea": {"seoul"},
        "canada": {
            "ontario",
            "alberta",
            "novascotia",
            "newfoundland",
            "newbrunswick",
            "britishcolumbia",
        },
        "miyanma": {"burma"},
        "newzealand": {"auckland", "nz", "otago"},
        "spain": {
            "andalucia",
            "pontevedra",
            "gipuzkoa",
            "lleida",
            "catalunyaspain",
            "galiza",
            "espaa",
        },
        "germany": {"niedersachsen", "deutschland"},
        "brazil": {"disritofederal"},
        "switzerland": {"lasuisse"},
        "italy": {"veneziagiulia", "ferrara", "italia"},
        "australia": {"nsw", "queensland", "newsouthwales"},
        "belgium": {"labelgique", "bergued"},
        "uruguay": {"urugua"},
        "panama": {"republicofpanama"},
    }

    country_del_list = {
        "c",
        "space",
        "universe",
        "unknown",
        "quit",
        "tdzimi",
        "universe",
        "tn",
        "unknown",
        "space",
        "c",
        "franciscomorazan",
        "petrolwarnation",
        "ineurope",
        "hereandthere",
        "faraway",
    }

    # 잘못된 나라 고치기
    for right_country, wrong_country in country_fix_dict.items():
        users.loc[
            users["location_country"].isin(wrong_country), "location_country"
        ] = right_country

    # 아예 잘못된 놈들은 삭제
    users = users[~users["location_country"].isin(country_del_list)]

    # 이제 남은 것들은 그냥 버리기
    users = users.dropna(
        subset=["location_state", "location_city", "location_country"]
    ).reset_index(drop=True)

    # age에 대한 처리
    def age_map(x):
        # 60 대 이상
        if x >= 60:
            return 6
        # 0 ~ 19 까지 한 묶음
        elif 0 <= x < 10:
            return 1
        else:
            return x // 10

    users.loc[~users["age"].isna(), "age"] = (
        users.loc[~users["age"].isna(), "age"].apply(age_map).astype(int)
    )

    return users


def process_item_data(books: pd.DataFrame) -> pd.DataFrame:
    '''
    # summary 삭제
    books.drop(["summary"], axis=1, inplace=True)

    # year_of_publication 타입 변환
    books["year_of_publication"] = books["year_of_publication"].astype(int)

    def year_of_publication_map(y):
        if y < 1900:
            return 0
        elif y < 1950:
            return 1
        else:
            return y // 20

    books["year_of_publication"] = books["year_of_publication"].apply(
        year_of_publication_map
    )

    # category 대괄호 풀고 소문자로
    books.loc[books[books["category"].notnull()].index, "category"] = books[
        books["category"].notnull()
    ]["category"].apply(lambda x: re.sub("[\W_]+", " ", x).strip())
    books["category"] = books["category"].str.lower()

    # category high 만들기
    books["category_high"] = books["category"].copy()

    # 분류해 낼 카테고리
    categories = [
        "history",
        "fiction",
        "nonfiction",
        "physics",
        "adventure",
        "fiction",
        "nonfiction",
        "science",
        "science fiction",
        "jouvenile fiction",
        "jouvenille nonfiction",
        "disease",
        "mathemat",
        "agricult",
        "business",
        "poetry",
        "drama",
        "literary",
        "travel",
        "motion picture",
        "children",
        "literature",
        "electronic",
        "humor",
        "computer",
        "house",
        "family",
        "architect",
        "camp",
        "language",
        "comic",
        "sports",
        "novel",
    ]

    # 얘는 여러개 인 경우도 포함하려고.
    groupings = {
        "hobby": ["crafts", "hobbies", "hobby", "garden"],
        "art/media": ["art", "photograph", "music"],
        "house/cook": ["cook", "house"],
        "business": ["economics", "business"],
        "religion": ["christian", "bible", "religion"],
        "animal": ["animal", "bird"],
        "social science": ["philosophy", "psycholog", "social", "sociology"],
        "criminal": ["criminal", "homicide"],
        "eco": ["ecology", "environment", "nature"],
    }

    # 상위 카테고리 묶기
    for category in categories:
        books.loc[
            books["category"].str.contains(category, na=False), "category_high"
        ] = category
    # grouping 하기
    for high, low in groupings.items():
        low = "|".join(low)
        books.loc[books["category"].str.contains(low, na=False), "category_high"] = high

    # 카테고리 high nan 처리
    books["category_high"] = books["category_high"].fillna("0")

    # 5개 미만의 카테고리는 others로 처리
    books_count = books["category_high"].value_counts()
    temp = books_count[books["category_high"].values] < 5
    books.loc[temp.values, "category_high"] = "others"

    # 제목 깨진거 처리
    def text_preprocessing_func(text: str) -> str:
        """
        깨진 문자를 변환하는 함수
        """
        text = text.replace("Ã?Â©", "e")  # 원래는 é인데 걍 e로 메움
        text = text.replace("Ã©", "e")
        text = text.replace("Ã?Â?", "e")  # 원래는 é인데 걍 e로 메움
        text = text.lower()
        return text

    books["book_title"] = books["book_title"].apply(text_preprocessing_func)

    # publisher 처리 EDA 기본 코드 활용

    # 기본 텍스트 전처리
    # 오래걸려서 잠시 주석처리
    books["publisher"] = books["publisher"].apply(text_preprocessing_func)

    publisher_dict = (books["publisher"].value_counts()).to_dict()
    publisher_count_df = pd.DataFrame(
        list(publisher_dict.items()), columns=["publisher", "count"]
    )

    publisher_count_df = publisher_count_df.sort_values(by=["count"], ascending=False)

    modify_list = publisher_count_df[publisher_count_df["count"] > 1].publisher.values
    for publisher in tqdm(modify_list):
        try:
            number = (
                books[books["publisher"] == publisher]["isbn"]
                .apply(lambda x: x[:4])
                .value_counts()
                .index[0]
            )
            right_publisher = (
                books[books["isbn"].apply(lambda x: x[:4]) == number]["publisher"]
                .value_counts()
                .index[0]
            )
            books.loc[
                books[books["isbn"].apply(lambda x: x[:4]) == number].index, "publisher"
            ] = right_publisher
        except:
            pass

    # 이 부분 아직 제대로 못읽어봄
    ########
    isbn_dict = {}
    isbn_dict = {
        books["language"][idx]: [isbn[:3]]
        if books["language"][idx] not in isbn_dict.keys()
        else isbn_dict[books["language"][idx]].append(isbn[:2])
        for idx, isbn in enumerate(books["isbn"])
    }
    isbn_code = {
        "0": "english",
        "1": "english",
        "2": "franch",
        "3": "german",
        "4": "japan",
        "5": "russia",
        "7": "china",
        "65": "brazil",
        "80": "czecho",
        "81": "india",
        "82": "norway",
        "83": "poland",
        "84": "espanol",
        "85": "brazil",
        "86": "yugoslavia",
        "87": "danish",
        "88": "italy",
        "89": "korean",
        "90": "netherlands",
        "91": "sweden",
        "92": "international ngo",
        "93": "inida",
        "94": "netherlands",
        "600": "iran",
        "601": "kazakhstan",
        "602": "indonesia",
        "603": "saudi arabia",
        "604": "vietnam",
        "605": "turkey",
        "606": "romania",
        "607": "mexico",
        "608": "north macedonia",
        "609": "lithuania",
        "611": "thailand",
        "612": "peru",
        "613": "mauritius",
        "614": "lebanon",
        "615": "hungary",
        "616": "thailand",
        "617": "ukraine",
        "618": "greece",
        "619": "bulgaria",
        "620": "mauritius",
        "621": "phillippines",
        "622": "iran",
        "623": "indonesia",
        "624": "sri lanka",
        "625": "turkey",
        "626": "taiwan",
        "627": "pakistan",
        "628": "colombia",
        "629": "malaysia",
        "630": "romania",
        "950": "argentina",
        "951": "finland",
        "952": "finland",
        "953": "croatia",
        "954": "bulgaria",
        "955": "sri lanka",
        "956": "chile",
        "957": "taiwan",
        "958": "colombia",
        "959": "cuba",
        "960": "greece",
        "961": "slovenia",
        "962": "hong kong",
        "963": "hungary",
        "964": "iran",
        "965": "israel",
        "966": "urkaine",
        "967": "malaysia",
        "968": "mexico",
        "969": "pakistan",
        "970": "mexico",
        "971": "phillippines",
        "972": "portugal",
        "973": "romania",
        "974": "thailand",
        "975": "turkey",
        "976": "caribbean community",
        "977": "egypt",
        "978": "nigeria",
        "979": "indonesia",
        "980": "venezuela",
        "981": "singapore",
        "982": "south pacific",
        "983": "malaysia",
        "984": "bangladesh",
        "985": "velarus",
        "986": "taiwan",
        "987": "argentina",
        "988": "hong kong",
        "989": "portugal",
        "9960": "saudi arabia",
        "9963": "cyprus",
        "9968": "costa rica",
        "9971": "singapore",
        "9972": "peru",
        "9974": "uruguay",
        "9976": "tanzania",
        "9977": "costa rica",
        "9979": "iceland",
        "9986": "lithuania",
        "99903": "mauritius",
        "99905": " bolivia",
        "99909": "malta",
        "99912": "botswana",
        "99920": "andorra",
        "99928": "georgia",
        "99935": "haiti",
        "99936": "bhutan",
        "99942": "armenia",
        "99943": "albania",
        "99974": "bolivia",
        "99975": "mongolia",
        "99989": "paraguay",
    }
    check_list = []
    books["isbn_country"] = "na"
    for idx in tqdm(range(len(books))):
        isbn = books["isbn"][idx][:5]
        if isbn[0] in isbn_code.keys():
            books.at[idx, "isbn_country"] = isbn_code[isbn[0]]
        elif isbn[:2] in isbn_code.keys():
            books.at[idx, "isbn_country"] = isbn_code[isbn[0:2]]
        elif isbn[:3] in isbn_code.keys():
            books.at[idx, "isbn_country"] = isbn_code[isbn[0:3]]
        elif isbn[:4] in isbn_code.keys():
            books.at[idx, "isbn_country"] = isbn_code[isbn[:4]]
        elif isbn[:] in isbn_code.keys():
            books.at[idx, "isbn_country"] = isbn_code[isbn[:]]
        else:
            check_list.append(isbn)
    books.loc[books["isbn_country"] == "na", "isbn_country"] = "english"

    books["book_author"] = books["book_author"].apply(text_preprocessing_func)

    ########'''

    return books


def process_tree_data(
    users: pd.DataFrame,
    books: pd.DataFrame,
    rating_train: pd.DataFrame,
    rating_test: pd.DataFrame,
) -> tuple:

    ratings = pd.concat([rating_train, rating_test]).reset_index(drop=True)

    # 활용할 context 목록
    context_cols = [
        "age",
        "location_state",
        "location_country",
        "book_author",
        "year_of_publication",
        "publisher",
        "category_high",
        "isbn_country",
    ]

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


def tree_data_load(args):

    ######################## DATA LOAD
    users = pd.read_csv(args.DATA_PATH + "users_processed.csv")
    users = process_user_data(users)

    books = pd.read_csv(args.DATA_PATH + "books.csv")
    # 오래 걸려서 그냥 일단 저장
    # books = process_item_data(books)
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

    train["user_id"] = train["user_id"].map(user2idx)
    sub["user_id"] = sub["user_id"].map(user2idx)
    test["user_id"] = test["user_id"].map(user2idx)
    users["user_id"] = users["user_id"].map(user2idx)

    train["isbn"] = train["isbn"].map(isbn2idx)
    sub["isbn"] = sub["isbn"].map(isbn2idx)
    test["isbn"] = test["isbn"].map(isbn2idx)
    books["isbn"] = books["isbn"].map(isbn2idx)

    idx, context_train, context_test = process_tree_data(users, books, train, test)
    field_dims_list = [len(user2idx), len(isbn2idx)]
    for k, v in idx.items():
        field_dims_list.append(len(v))

    field_dims = np.array(
        field_dims_list,
        dtype=np.uint32,
    )

    data = {
        "train": context_train,
        "test": context_test.drop(["rating"], axis=1),
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


def tree_data_split(args, data):
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


def tree_data_loader(args, data):
    train_dataset = TensorDataset(
        torch.LongTensor(data["X_train"].values),
        torch.LongTensor(data["y_train"].values),
    )
    valid_dataset = TensorDataset(
        torch.LongTensor(data["X_valid"].values),
        torch.LongTensor(data["y_valid"].values),
    )
    test_dataset = TensorDataset(torch.LongTensor(data["test"].values))

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
