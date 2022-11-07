import pandas as pd
import numpy as np
import scipy.stats as ss
from tqdm import tqdm
import re
from ..utils import (
    text_preprocessing_func,
    isbn_code,
    country_del_list,
    country_fix_dict,
    fiction,
    unclassified,
)


class Dataset:
    def __init__(
        self, users: pd.DataFrame, books: pd.DataFrame, train_ratings: pd.DataFrame
    ) -> None:
        self.users = users
        self.books = books
        self.train_ratings = train_ratings

    def preprocess_users(self):
        self.preprocess_location()
        self.preprocess_age()

    def preprocess_books(
        self, drops=["book_title", "img_url", "language", "summary", "img_path"]
    ):
        self.preprocess_author()
        self.preprocess_year()
        self.preprocess_publisher()
        self.preprocess_language()
        self.preprocess_category()
        self.drop_book_columns(drops)

    def preprocess_ratings(self):
        user_count = (
            self.train_ratings.groupby("user_id").count()["isbn"].to_dict()
        )  # user_id 항목별로 묶을 건데, isbn의 수를 세자 즉, 한 유저가 평가한 isbn 수?
        for i in range(len(self.train_ratings)):
            self.train_ratings.at[i, "count"] = user_count[
                self.train_ratings["user_id"][i]
            ]  # at은 row이름, column 이름으로 해당 데이터에 접근, self.train_ratings에 count라는 값이 생김
        self.train_ratings = self.train_ratings[
            self.train_ratings["count"] > 2
        ].reset_index(drop=True)
        self.train_ratings = self.train_ratings.drop("count", axis=1)

    def df_to_csv(self):
        self.users.to_csv("users_preprocessed.csv", index=False)
        self.books.to_csv("books_preprocessed.csv", index=False)
        self.train_ratings.to_csv("train_ratings_preprocessed.csv", index=False)

    def preprocess_location(self):
        """
        self.users의 location을 나누고 정리하는 함수
        1. city, state, country로 나누고
        2. country, city가 같다면 state도 같다는 점을 이용해 state를 채움
        3. country, state, city 중 하나라도 na인 row를 삭제
        4. country 이상한 애들 직접 수정
        """

        def fix_location(location: str, right_location: pd.Series) -> str:
            try:
                if right_location.loc[location]:
                    return location
                else:
                    return right_location.index[0]
            except:
                return right_location.index[0]

        # (1) city, state, country를 나눔
        self.users["location"] = self.users["location"].str.replace(
            r"[^0-9a-zA-Z,]", ""
        )  # 특수문자 제거

        self.users["location_city"] = self.users["location"].apply(
            lambda x: x.split(",")[0].strip()
        )
        self.users["location_state"] = self.users["location"].apply(
            lambda x: x.split(",")[1].strip()
        )
        self.users["location_country"] = self.users["location"].apply(
            lambda x: x.split(",")[2].strip()
        )

        self.users = self.users.replace(
            "na", np.nan
        )  # 특수문자 제거로 n/a가 na로 바뀌게 되었습니다. 따라서 이를 컴퓨터가 인식할 수 있는 결측값으로 변환합니다.
        self.users = self.users.replace(
            "", np.nan
        )  # 일부 경우 , , ,으로 입력된 경우가 있었으므로 이런 경우에도 결측값으로 변환합니다.

        modify_location = self.users[
            (self.users["location_country"].isna())
            & (self.users["location_city"].notnull())
        ]["location_city"].values
        location = (
            self.users[
                (self.users["location"].str.contains("seattle"))
                & (self.users["location_country"].notnull())
            ]["location"]
            .value_counts()
            .index[0]
        )

        location_list = []
        for location in modify_location:
            try:
                right_location = (
                    self.users[
                        (self.users["location"].str.contains(location))
                        & (self.users["location_country"].notnull())
                    ]["location"]
                    .value_counts()
                    .index[0]
                )
                location_list.append(right_location)
            except:
                pass

        for location in location_list:
            self.users.loc[
                self.users[self.users["location_city"] == location.split(",")[0]].index,
                "location_state",
            ] = location.split(",")[1]
            self.users.loc[
                self.users[self.users["location_city"] == location.split(",")[0]].index,
                "location_country",
            ] = location.split(",")[2]

        # (2) country, city가 같다면 state도 같다는 점을 이용해 state를 채움
        modify_city = self.users["location_city"].unique()
        for city in tqdm(modify_city):
            try:
                right_state = self.users[
                    (self.users["location"].str.contains(city))
                    & (self.users["location_state"].notnull())
                ]["location_state"].value_counts()
                right_country = self.users[
                    (self.users["location"].str.contains(city))
                    & (self.users["location_country"].notnull())
                ]["location_country"].value_counts()

                right_state = right_state / right_state.sum() > 0.08
                right_country = right_country / right_country.sum() > 0.08

                modify_index = self.users.loc[self.users["location_city"] == city].index

                self.users.loc[modify_index, "location_state"] = self.users.loc[
                    modify_index, "location_state"
                ].apply(fix_location, args=(right_state,))
                self.users.loc[modify_index, "location_country"] = self.users.loc[
                    modify_index, "location_country"
                ].apply(fix_location, args=(right_country,))

            except:
                pass

        self.users["location_country"] = self.users["location_country"].fillna("others")

        del_idx = []
        for idx, row in enumerate(self.users["location_country"]):
            for key, value in country_fix_dict.items():
                if row in value:
                    self.users.at[idx, "location_country"] = key
            if row in country_del_list:
                del_idx.append(idx)

        for i in del_idx:
            self.users.at[i, "location_country"] = "others"

        self.users = self.users.drop(
            ["location", "location_city", "location_state"], axis=1
        )

        users_country = [
            "usa",
            "canada",
            "germany",
            "unitedkingdom",
            "australia",
            "spain",
            "france",
        ]  # italy 빠짐
        for i in range(len(self.users)):
            if self.users.at[i, "location_country"] not in users_country:
                self.users.at[i, "location_country"] = "others"

    def preprocess_age(self):
        self.users = self.users.drop(["age"], axis=1)

    # book

    def drop_book_columns(self, columns):
        self.books = self.books.drop(columns, axis=1)

    def preprocess_author(self):
        # 기본 텍스트 전처리
        self.books["book_author"] = self.books["book_author"].apply(
            text_preprocessing_func
        )
        # 숫자 삭제
        self.books["book_author"] = self.books["book_author"].apply(
            lambda x: re.sub("\d", "", x).strip()
        )
        # not applicable은 na로 넣어줌
        self.books.loc[
            self.books[self.books["book_author"].str.contains("not applicable")].index,
            "book_author",
        ] = "na"

    def preprocess_year(self):
        def year_map(x: int) -> int:
            x = int(x)
            if x > 2000:
                return 1
            elif x >= 1990:
                return 2
            elif x >= 1980:
                return 3
            elif x >= 1970:
                return 4
            else:
                return 5

        self.books["year_of_publication"] = self.books["year_of_publication"].astype(
            int
        )
        self.books["year_of_publication"] = self.books["year_of_publication"].apply(
            year_map
        )

    def preprocess_publisher(self):
        # publisher 처리

        self.books["publisher"] = self.books["publisher"].apply(text_preprocessing_func)
        publisher_dict = (self.books["publisher"].value_counts()).to_dict()
        publisher_count_df = pd.DataFrame(
            list(publisher_dict.items()), columns=["publisher", "count"]
        )

        publisher_count_df = publisher_count_df.sort_values(
            by=["count"], ascending=False
        )

        modify_list = publisher_count_df[
            publisher_count_df["count"] > 1
        ].publisher.values
        for publisher in modify_list:
            try:
                number = (
                    self.books[self.books["publisher"] == publisher]["isbn"]
                    .apply(lambda x: x[:4])
                    .value_counts()
                    .index[0]
                )
                right_publisher = (
                    self.books[self.books["isbn"].apply(lambda x: x[:4]) == number][
                        "publisher"
                    ]
                    .value_counts()
                    .index[0]
                )
                self.books.loc[
                    self.books[
                        self.books["isbn"].apply(lambda x: x[:4]) == number
                    ].index,
                    "publisher",
                ] = right_publisher
            except:
                pass

    def preprocess_language(self):
        check_list = []
        self.books["isbn_country"] = "na"
        for idx in range(len(self.books)):
            isbn = self.books["isbn"][idx][:5]
            if isbn[0] in isbn_code.keys():
                self.books.at[idx, "isbn_country"] = isbn_code[isbn[0]]
            elif isbn[:2] in isbn_code.keys():
                self.books.at[idx, "isbn_country"] = isbn_code[isbn[0:2]]
            elif isbn[:3] in isbn_code.keys():
                self.books.at[idx, "isbn_country"] = isbn_code[isbn[0:3]]
            elif isbn[:4] in isbn_code.keys():
                self.books.at[idx, "isbn_country"] = isbn_code[isbn[:4]]
            elif isbn[:] in isbn_code.keys():
                self.books.at[idx, "isbn_country"] = isbn_code[isbn[:]]
            else:
                check_list.append(isbn)
        self.books[self.books["isbn_country"] == "na"]["isbn_country"] = "english"
        books_language = ["english", "german", "franch", "espanol"]
        for i in range(len(self.books)):
            if self.books.at[i, "isbn_country"] not in books_language:
                self.books.at[i, "isbn_country"] = "others"

    def preprocess_category(self):
        self.books.loc[
            self.books[self.books["category"].notnull()].index, "category"
        ] = self.books[self.books["category"].notnull()]["category"].apply(
            lambda x: re.sub("[\W_]+", " ", x).strip()
        )  # 일단 category에서 대괄호 밖으로 빼기
        self.books["category"] = self.books["category"].str.lower()  # 소문자로 바꾸기

        self.books["category_high"] = self.books[
            "category"
        ].copy()  # category_high로 category를 복사

        self.books["category_high"] = self.books["category_high"].fillna(
            "unclassified"
        )  # 안 채워진 건 미분류 항목으로 넣기
        self.books["year_of_publication"] = self.books["year_of_publication"].astype(
            int
        )

        self.books.loc[
            self.books[self.books["category"].notnull()].index, "category"
        ] = self.books[self.books["category"].notnull()]["category"].apply(
            lambda x: re.sub("[\W_]+", " ", x).strip()
        )  # 일단 category에서 대괄호 밖으로 빼기
        self.books["category"] = self.books["category"].str.lower()  # 소문자로 바꾸기

        self.books["category_high"] = self.books[
            "category"
        ].copy()  # category_high로 category를 복사

        self.books["category_high"] = self.books["category_high"].fillna(
            "unclassified"
        )  # 안 채워진 건 미분류 항목으로 넣기

        for small in fiction:
            self.books.loc[
                self.books[self.books["category"].str.contains(small, na=False)].index,
                "category_high",
            ] = "fiction"

        for small in unclassified:
            self.books.loc[
                self.books[self.books["category"].str.contains(small, na=False)].index,
                "category_high",
            ] = "unclassified"

        self.books.loc[
            self.books[self.books["category"].str.contains("nonfic", na=False)].index,
            "category_high",
        ] = "nonfiction"

        self.books.loc[
            ~self.books["category_high"].isin(["fiction", "unclassified"]),
            "category_high",
        ] = "nonfiction"
