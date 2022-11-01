import pandas as pd
import numpy as np
import scipy.stats as ss
from tqdm import tqdm
import re

# loading data
users = pd.read_csv("./data/users.csv")
books = pd.read_csv("./data/books.csv", encoding="utf-8")
ratings = pd.read_csv("./data/train_ratings.csv")
test = pd.read_csv("./data/test_ratings.csv")
# -----------------------------------------------
# functions


def text_preprocessing_func(text: str) -> str:
    """
    깨진 문자를 변환하고 특수문자를 삭제하는 함수
    """
    text = text.replace("Ã?Â©", "e")  # 원래는 é인데 걍 e로 메움
    text = text.replace("Ã©", "e")
    text = text.replace("Ã?Â?", "e")  # 원래는 é인데 걍 e로 메움
    text = text.lower()
    text = text.replace("ã", "a")
    text = text.replace("\xa0", " ")
    text = text.replace("â", "a")
    text = re.sub(r"[^\w\d ]", "", text)
    del_list = ["³", "º", "ª", "¼", "µ", "¹", "²", "½"]
    for del_word in del_list:
        text = text.replace(del_word, "")
    return text


# USERS ----------------------------------------
def fix_location(location: str, right_location: pd.Series) -> str:
    try:
        if right_location.loc[location]:
            return location
        else:
            return right_location.index[0]
    except:
        return right_location.index[0]


def location_preprocessing_func(users: pd.DataFrame) -> pd.DataFrame:
    """
    users의 location을 나누고 정리하는 함수
    1. city, state, country로 나누고
    2. country, city가 같다면 state도 같다는 점을 이용해 state를 채움
    3. country, state, city 중 하나라도 na인 row를 삭제
    4. country 이상한 애들 직접 수정
    """

    # (1) city, state, country를 나눔
    users["location"] = users["location"].str.replace(r"[^0-9a-zA-Z,]", "")  # 특수문자 제거

    users["location_city"] = users["location"].apply(lambda x: x.split(",")[0].strip())
    users["location_state"] = users["location"].apply(lambda x: x.split(",")[1].strip())
    users["location_country"] = users["location"].apply(
        lambda x: x.split(",")[2].strip()
    )

    users = users.replace(
        "na", np.nan
    )  # 특수문자 제거로 n/a가 na로 바뀌게 되었습니다. 따라서 이를 컴퓨터가 인식할 수 있는 결측값으로 변환합니다.
    users = users.replace(
        "", np.nan
    )  # 일부 경우 , , ,으로 입력된 경우가 있었으므로 이런 경우에도 결측값으로 변환합니다.

    modify_location = users[
        (users["location_country"].isna()) & (users["location_city"].notnull())
    ]["location_city"].values
    location = (
        users[
            (users["location"].str.contains("seattle"))
            & (users["location_country"].notnull())
        ]["location"]
        .value_counts()
        .index[0]
    )

    location_list = []
    for location in modify_location:
        try:
            right_location = (
                users[
                    (users["location"].str.contains(location))
                    & (users["location_country"].notnull())
                ]["location"]
                .value_counts()
                .index[0]
            )
            location_list.append(right_location)
        except:
            pass

    for location in location_list:
        users.loc[
            users[users["location_city"] == location.split(",")[0]].index,
            "location_state",
        ] = location.split(",")[1]
        users.loc[
            users[users["location_city"] == location.split(",")[0]].index,
            "location_country",
        ] = location.split(",")[2]

    # (2) country, city가 같다면 state도 같다는 점을 이용해 state를 채움
    modify_city = users["location_city"].unique()
    for city in tqdm(modify_city):
        try:
            right_state = users[
                (users["location"].str.contains(city))
                & (users["location_state"].notnull())
            ]["location_state"].value_counts()
            right_country = users[
                (users["location"].str.contains(city))
                & (users["location_country"].notnull())
            ]["location_country"].value_counts()

            right_state = right_state / right_state.sum() > 0.08
            right_country = right_country / right_country.sum() > 0.08

            modify_index = users.loc[users["location_city"] == city].index

            users.loc[modify_index, "location_state"] = users.loc[
                modify_index, "location_state"
            ].apply(fix_location, args=(right_state,))
            users.loc[modify_index, "location_country"] = users.loc[
                modify_index, "location_country"
            ].apply(fix_location, args=(right_country,))

        except:
            pass

    # (3) na인 row를 삭제
    users = users.dropna(
        subset=["location_state", "location_city", "location_country"]
    ).reset_index(drop=True)

    # (4) country 이상한 애들 직접 수정
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
    country_del_list = [
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
    ]
    del_idx = []
    for idx, row in enumerate(users["location_country"]):
        for key, value in country_fix_dict.items():
            if row in value:
                users.at[idx, "location_country"] = key
        if row in country_del_list:
            del_idx.append(idx)
    users = users.drop(del_idx, axis=0).reset_index(drop=True)

    return users


# books ----------------------------------------
def category_preprocessing_func(books: pd.DataFrame) -> pd.DataFrame:
    """
    category를 정리하는 함수
    1. category에서 대괄호 밖으로 빼기
    2. category high를 만듦
    3. category high에서 10% 미만인 11개를 Unclassified로 추가 분류
    """
    # 1. category에서 대괄호 밖으로 빼기
    books.loc[books[books["category"].notnull()].index, "category"] = books[
        books["category"].notnull()
    ]["category"].apply(lambda x: re.sub("[\W_]+", " ", x).strip())
    books["category"] = books["category"].str.lower()
    # 2. category high를 만들기
    books["category_high"] = books["category"].copy()
    groupings = {
        "Fiction": ["fiction", "ficti"],  # 너무 넓으니 맨 위로 빼자
        "Literature & Poem": ["liter", "poem", "poetry", "novel", "sonnet"],
        "Science & Math": [
            "science",
            "math",
            "logy",
            "chemis",
            "physics",
            "electron",
        ],  # science, logy 범위가 너무 넓으니 맨 위로
        "Parenting & Relationships": [
            "baby",
            "babies",
            "parent",
            "family",
            "friend",
            "tionship",
            "brother",
            "sister",
            "families",
            "friendship",
            "mother",
            "father",
        ],  # 좀 큼
        "Medical Books": ["medi", "psycho"],  # psy의 세분화 가능
        "Adventure": ["adventu"],
        "Animal & Nature": [
            "animal",
            "ecolo",
            "plant",
            "nature",
            "cat",
            "dog",
            "pets",
            "bird",
            "bear",
            "horse",
            "frog",
            "duck",
            "rabbit",
            "dolphin",
            "mice",
            "deer",
            "panda",
            "kangaroo",
            "lizzard",
            "gorilla",
            "chimpangee",
            "bat",
            "insect",
        ],
        "Arts & Photography": [
            "art",
            "photo",
            "drawing",
            "picture",
        ],  # art는 겹치는 글자가 너무 많음
        "Authors": ["authors"],
        "Biographies & Memoirs": ["biog", "memo"],
        "Business & Money": ["busi", "money", "econo", "finance"],
        "Calendars": ["calen"],
        "Children's Books": ["child", "baby"],
        "Christian Books & Bibles": ["christi", "bible"],  # 크리스마스때매
        "Christmas": ["christma"],
        "Comics & Graphic Novels": ["comics", "graphic novel"],
        "Computers & Technology": ["computer", "techno", "archi"],
        "Cookbooks, Food & Wine": ["cook", "wine", "food"],
        "Countries & Cities": [
            "united states",
            "russia",
            "france",
            "africa",
            "china",
            "japan",
            "egypt",
            "germany",
            "ireland",
            "california",
            "berline",
            "london",
            "new york",
            "canada",
            "chile",
            "italy",
            "europe",
            "australia",
            "great britain",
            "arizona",
            "chicago",
            "netherlands",
            "calif",
            "mexico",
            "colombia",
            "greece",
            "florida",
            "algeria",
            "new zealand",
            "austria",
            "denmark",
            "washington",
            "india",
            "england",
            "brazil",
        ],
        "Crafts, Hobbies & Home": ["crafts", "hobb", "home", "house", "garden"],
        "Crime & Murder": [
            "crime",
            "murder",
            "criminal",
            "homicide",
            "mafia",
            "gang",
            "drug",
        ],
        "Critic": ["critic"],
        "Education & Teaching": ["educa", "teach"],
        "Drama": ["drama"],
        "Design": ["design"],
        "Engineering & Transportation": ["engine", "transp"],
        "Encyclopedia & Dictionary": ["encyclo", "dictiona", "vocabulary"],
        "Essay": ["essay"],
        "Health, Fitness & Dieting": ["health", "fitness", "diet"],
        "History": ["histo", "war"],
        "Humor & Entertainment": ["humor", "entertai", "comed", "game", "comic"],
        "Law": ["law"],
        "Language": ["language"],
        "LGBTQ+ Books": ["lesbian", "gay", "bisex"],
        "Mystery, Thriller & Suspense": [
            "myste",
            "thril",
            "suspen",
            "horror",
            "occult",
        ],
        "Music & Dance": ["music", "dance", "instrument", "ballet", "classic"],
        "Movie": [
            "motion pictur",
            "actor",
            "actres",
            "acting",
            "cinema",
            "theater",
            "director",
            "television",
        ],
        "Politics": ["politic", "president"],
        "Philosophy": ["philoso"],
        "Reference": ["reference"],
        "Religion & Spirituality": [
            "religi",
            "buddh",
            "spirit",
            "god",
            "prayer",
            "belief",
            "doubt",
        ],
        "Romance": ["romance"],
        "Science Fiction & Fantasy": [
            "imagin",
            "science fiction",
            "fantasy",
            "fairy",
            "fairies",
            "vampire",
            "epidemic",
            "ghost",
            "alien",
            "supernatural",
            "magic",
            "dragons",
            "elves",
            "angel",
            "devil",
        ],
        "Short story": ["short"],
        "Social Science": [
            "social",
            "ethic",
            "communism",
            "capitalism",
            "generation",
            "culture",
        ],
        "Self-Help": ["self"],  # self 검색시 모두 자기계발 관련
        "Study": ["test", "school", "examina", "study aids", "college"],
        "Sports & Outdoors": [
            "exerc",
            "sport",
            "outdoor",
            "baseball",
            "soccer",
            "hockey",
            "cricket",
            "basketball",
            "footbal",
        ],
        "Teen & Young Adult": ["teen", "adol", "juven"],  # nonfiction이란 말은 청소년 관련뿐
        "Travel": ["travel"],
        "Women": ["women"],
    }

    # grouping 하기
    for key, value in groupings.items():
        for val in value:
            books.loc[
                books[books["category"].str.contains(val, na=False)].index,
                "category_high",
            ] = key

    # count column 새로 만들어줌
    books_count = books["category_high"].value_counts().to_dict()
    for i in range(len(books)):
        books.at[i, "count"] = books_count[books.loc[i, "category_high"]]

    # 추가적으로 11개 미만인 애들은 미분류로 추가 편입 -> 왜냐면 10퍼센트가 11개라서
    for i in range(len(books)):
        if books.at[i, "count"] < 11:
            books.at[i, "category_high"] = "Unclassified"

    return books


def book_title_preprocessing_func(books: pd.DataFrame) -> pd.DataFrame:
    """
    book_title을 전처리 하는 코드
    1. 글자 깨진거 수정, 소문자처리
    """
    books["book_title"] = books["book_title"].apply(text_preprocessing_func)
    return books


def book_publisher_preprocessing_func(books: pd.DataFrame) -> pd.DataFrame:
    books["publisher"] = books["publisher"].apply(text_preprocessing_func)
    publisher_dict = (books["publisher"].value_counts()).to_dict()
    publisher_count_df = pd.DataFrame(
        list(publisher_dict.items()), columns=["publisher", "count"]
    )

    publisher_count_df = publisher_count_df.sort_values(by=["count"], ascending=False)

    modify_list = publisher_count_df[publisher_count_df["count"] > 1].publisher.values
    for publisher in modify_list:
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


def book_language_preprocessing_func(books: pd.DataFrame) -> pd.DataFrame:
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
    for idx in range(len(books)):
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
    books[books["isbn_country"] == "na"]["isbn_country"] = "english"
    return books

def books_author_preprocessing_func(books: pd.DataFrame) -> pd.DataFrame :
    books['book_author'] = books['author'].apply(text_preprocessing_func)
    return books
# ------------------------------------------------
