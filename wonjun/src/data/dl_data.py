import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset
import re

def text_preprocessing_func(text : str ) -> str :
    """
    깨진 문자를 변환하는 함수
    """
    text = text.replace('Ã?Â©','e') # 원래는 é인데 걍 e로 메움
    text = text.replace('Ã©', 'e')
    text = text.replace('Ã?Â?','e') # 원래는 é인데 걍 e로 메움
    text = text.lower()
    return text

def dl_data_load(args):

    ######################## DATA LOAD
    users = pd.read_csv(args.DATA_PATH + 'users.csv')
    books = pd.read_csv(args.DATA_PATH + 'books.csv', encoding = 'utf-8')
    train = pd.read_csv(args.DATA_PATH + 'train_ratings.csv')
    test = pd.read_csv(args.DATA_PATH + 'test_ratings.csv')
    sub = pd.read_csv(args.DATA_PATH + 'sample_submission.csv')

    #-----------------------------user---------------------------------
    # 1 rating user cut
    user_count = train.groupby('user_id').count()['isbn'].to_dict()
    for i in range(len(train)):
        train.at[i, 'count'] = user_count[train['user_id'][i]]
    train = train[train['count']>1].reset_index(drop=True)
    train = train.drop("count", axis=1)

    # age cut
    users = users.drop('age', axis=1)

    # location split
    users['location'] = users['location'].str.replace(r'[^0-9a-zA-Z:,]', '') # 특수문자 제거

    users['location_city'] = users['location'].apply(lambda x: x.split(',')[0].strip())
    users['location_state'] = users['location'].apply(lambda x: x.split(',')[1].strip())
    users['location_country'] = users['location'].apply(lambda x: x.split(',')[2].strip())

    users = users.replace('na', np.nan) #특수문자 제거로 n/a가 na로 바뀌게 되었습니다. 따라서 이를 컴퓨터가 인식할 수 있는 결측값으로 변환합니다.
    users = users.replace('', np.nan) # 일부 경우 , , ,으로 입력된 경우가 있었으므로 이런 경우에도 결측값으로 변환합니다.

    modify_location = users[(users['location_country'].isna())&(users['location_city'].notnull())]['location_city'].values
    location = users[(users['location'].str.contains('seattle'))&(users['location_country'].notnull())]['location'].value_counts().index[0]

    location_list = []
    for location in modify_location:
        try:
            right_location = users[(users['location'].str.contains(location))&(users['location_country'].notnull())]['location'].value_counts().index[0]
            location_list.append(right_location)
        except:
            pass
        
    for location in location_list:
        users.loc[users[users['location_city']==location.split(',')[0]].index,'location_state'] = location.split(',')[1]
        users.loc[users[users['location_city']==location.split(',')[0]].index,'location_country'] = location.split(',')[2]

    # lacation na cut, location column cut
    users = users.dropna(subset=['location_state', 'location_city', 'location_country']).reset_index(drop=True)
    users = users.drop('location', axis=1)

    # country fix
    country_fix_dict = {'usa': {'oklahoma','districtofcolumbia', 'connecticut', 'worcester', 'aroostook', 'texas',  'kern', 'orangeco', 'unitedstatesofamerica', 'fortbend', 'alachua', 'massachusetts', 'arizona', 'austin', 'hawaii', 'ohio', 'camden', 'arkansas', 'minnesota', 'losestadosunidosdenorteamerica', 'us', 'usanow', 'northcarolina', 'maine', 'colorado','oklahoma', 'alabama', 'anystate', 'districtofcolumbia', 'unitedstaes', 'pender', 'newhampshire', 'unitedstates', 'missouri', 'idaho', 'ca', 'newyork','tennessee', 'stthomasi', 'dc', 'washington', 'illinois', 'california', 'michigan', 'iowa', 'maryland', 'newjersey', 'vanwert', 'oregon'},
                    'uk': {'alderney', 'wales',  'aberdeenshire', 'bermuda', 'nottinghamshire', 'scotland', 'usacurrentlylivinginengland', 'england', 'countycork', 'alderney', 'cambridgeshire', 'middlesex', 'northyorkshire', 'westyorkshire', 'cocarlow', 'sthelena'},
                    'japan': {'okinawa'},
                    'southkorea': {'seoul'},
                    'canada': {'ontario', 'alberta', 'novascotia', 'newfoundland', 'newbrunswick', 'britishcolumbia'},
                    'miyanma': {'burma'},
                    'newzealand': {'auckland', 'nz', 'otago'},
                    'spain': {'andalucia','pontevedra', 'gipuzkoa', 'lleida', 'catalunyaspain', 'galiza', 'espaa'},
                    'germany': {'niedersachsen', 'deutschland'},
                    'brazil': {'disritofederal'},
                    'switzerland': {'lasuisse'},
                    'italy': {'veneziagiulia', 'ferrara', 'italia'},
                    'australia': {'nsw', 'queensland', 'newsouthwales'},
                    'belgium': {'labelgique', 'bergued'},
                    'uruguay': {'urugua'},
                    'panama': {'republicofpanama'}
                   }
    country_del_list = ['c', 'space', 'universe', 'unknown', 'quit', 'tdzimi', 'universe', 'tn', 'unknown', 'space', 'c', 'franciscomorazan', 'petrolwarnation', 'ineurope', 'hereandthere', 'faraway']

    del_idx = []
    for idx, row in enumerate(users['location_country']):
        for key, value in country_fix_dict.items():
            if row in value:
                users.at[idx, 'location_country'] = key
        if row in country_del_list:
            del_idx.append(idx)
    users = users.drop(del_idx, axis=0).reset_index(drop=True)

    # location city cut
    # users = users.drop('location_state', axis=1) state cut
    users = users.drop('location_city', axis=1)

    #-----------------------------book---------------------------------
    # year of pub int
    books['year_of_publication'] = books['year_of_publication'].astype(int)
    # summary cut
    books = books.drop('summary', axis=1)
    # 일단 category에서 대괄호 밖으로 빼기
    books.loc[books[books['category'].notnull()].index, 'category'] = books[books['category'].notnull()]['category'].apply(lambda x: re.sub('[\W_]+',' ',x).strip())
    # 소문자로 바꾸기
    books['category'] = books['category'].str.lower()

    # category
    # 카테고리를 좀 더 큰 카테고리로 묶어주자
    # 지금 category_high는 분류 안 된 category와 동일한 상태
    books['category_high'] = books['category'].copy()

    groupings = {'Fiction': ['fiction'], # 너무 넓으니 맨 위로 빼자
             'Literature & Poem': ['liter', 'poem', 'poetry'],
             'Science & Math': ['science', 'math', 'logy'], # science, logy 범위가 너무 넓으니 맨 위로
             'Parenting & Relationships': ['baby', 'parent', 'family', 'tionship', 'brother', 'sister'], # 좀 큼
             'Medical Books': ['medi', 'psycho'], # psy의 세분화 가능
             'Animal & Nature': ['animal', 'ecolo', 'plant', 'nature'],
             
             'Arts & Photography': ['art', 'photo'], # art는 겹치는 글자가 너무 많음
             'Biographies & Memoirs': ['biog', 'memo'],
             'Business & Money': ['busi', 'money', 'econo'],
             'Calendars': ['calen'],
             'Children\'s Books': ['child', 'baby'],
             'Christian Books & Bibles': ['christi', 'bible'], #크리스마스때매
             'Comics & Graphic Novels': ['comics', 'graphic novel'],
             'Computers & Technology': ['computer', 'techno', 'archi'],
             'Cookbooks, Food & Wine': ['cook'],
             'Crafts, Hobbies & Home': ['crafts'],
             'Education & Teaching': ['educa', 'teach'],
             'Engineering & Transportation': ['engine', 'transp'],
             'Health, Fitness & Dieting': ['health', 'fitness', 'diet'],
             'History': ['histo'],
             
             'Humor & Entertainment': ['humor', 'entertai', 'comed', 'game'],
             'Law': ['law'],
             'LGBTQ+ Books': ['lesbian', 'gay', 'bisex'],
             'Mystery, Thriller & Suspense': ['myste', 'thril', 'suspen'],
             'Politics & Social Sciences': ['politic', 'social'],
             'Reference': ['reference'],
             'Religion & Spirituality': ['religi'],
             'Romance': ['romance'],
             'Science Fiction & Fantasy': ['science fiction', 'fantasy'],
             'Self-Help': ['self'], # self 검색시 모두 자기계발 관련
             'Sports & Outdoors': ['exerc','sport','outdoor'],
             'Teen & Young Adult': ['teen', 'adol', 'juven'], #nonfiction이란 말은 청소년 관련뿐
             'Test Preparation': ['test', 'school', 'examina'],
             'Travel': ['travel'],
                }
    # grouping 하기, 파편화된 카테고리를 대주제로 묶기
    for new_group, small in groupings.items():
        for s in small:
            books.loc[books[books['category'].str.contains(s, na = False)].index, 'category_high'] = new_group
    
    #category null 값 묶기
    books['category_new'] = books['category_high'].copy()
    books['category_new'] = books['category_new'].fillna('Unclassified')

    # title
    books['book_title'] = books['book_title'].apply(text_preprocessing_func)
    # publisher
    books['publisher'] = books['publisher'].apply(text_preprocessing_func)
    publisher_dict=(books['publisher'].value_counts()).to_dict()
    publisher_count_df= pd.DataFrame(list(publisher_dict.items()),columns = ['publisher','count'])

    publisher_count_df = publisher_count_df.sort_values(by=['count'], ascending = False)

    modify_list = publisher_count_df[publisher_count_df['count']>1].publisher.values
    for publisher in modify_list:
        try:
            number = books[books['publisher']==publisher]['isbn'].apply(lambda x: x[:4]).value_counts().index[0]
            right_publisher = books[books['isbn'].apply(lambda x: x[:4])==number]['publisher'].value_counts().index[0]
            books.loc[books[books['isbn'].apply(lambda x: x[:4])==number].index,'publisher'] = right_publisher
        except: 
            pass
    # language -> country
    isbn_dict = {}
    isbn_dict = { books['language'][idx] : [isbn[:3]] if books['language'][idx] not in isbn_dict.keys() else isbn_dict[books['language'][idx]].append(isbn[:2]) for idx, isbn in enumerate(books['isbn'])}
    isbn_code = {'0' : 'english', '1' : 'english', '2': 'franch', '3' : 'german', '4' : 'japan', '5' : 'russia', '7' : 'china',
             '65' : 'brazil', '80' : 'czecho', '81' : 'india', '82' : 'norway', '83' : 'poland', '84' : 'espanol', '85' : 'brazil', '86' : 'yugoslavia', '87' : 'danish', '88' : 'italy', '89' : 'korean', '90' : 'netherlands', '91' : 'sweden',
            '92' : 'international ngo', '93' : 'inida', '94' : 'netherlands', '600' : 'iran', '601' : 'kazakhstan', '602' : 'indonesia', '603' : 'saudi arabia', '604' : 'vietnam', '605' : 'turkey',
            '606' : 'romania', '607' : 'mexico', '608' : 'north macedonia', '609' : 'lithuania', '611' : 'thailand', '612' : 'peru', '613' : 'mauritius',
            '614' : 'lebanon', '615' : 'hungary', '616' : 'thailand', '617' : 'ukraine', '618' : 'greece', '619' : 'bulgaria', '620' : 'mauritius', '621' : 'phillippines',
            '622' : 'iran', '623' : 'indonesia', '624' : 'sri lanka', '625' : 'turkey', '626' : 'taiwan', '627' : 'pakistan', '628' : 'colombia', '629' : 'malaysia', '630' : 'romania',
            '950' : 'argentina', '951' : 'finland', '952' : 'finland', '953' : 'croatia', '954' : 'bulgaria', '955' : 'sri lanka',
            '956' : 'chile', '957' : 'taiwan', '958' : 'colombia', '959' : 'cuba', '960' : 'greece' , '961' : 'slovenia', '962' : 'hong kong',
            '963' : 'hungary', '964' : 'iran', '965' : 'israel', '966' : 'urkaine', '967' : 'malaysia', '968' : 'mexico', '969' : 'pakistan', '970' : 'mexico',
            '971' : 'phillippines', '972' : 'portugal', '973' : 'romania', '974' : 'thailand', '975' : 'turkey', '976' : 'caribbean community', '977' : 'egypt', '978' : 'nigeria', 
            '979' : 'indonesia', '980' : 'venezuela', '981' : 'singapore', '982' : 'south pacific', '983' : 'malaysia', '984' : 'bangladesh', '985' : 'velarus', '986' : 'taiwan',
            '987' : 'argentina', '988' : 'hong kong', '989' : 'portugal',
            '9960':'saudi arabia', '9963' : 'cyprus', '9968' : 'costa rica', '9971' : 'singapore', '9972' : 'peru', '9974' : 'uruguay',
            '9976' : 'tanzania', '9977' : 'costa rica', '9979' : 'iceland', '9986' : 'lithuania',
            '99903' : 'mauritius', '99905' : ' bolivia', '99909' : 'malta', '99912' : 'botswana', '99920' : 'andorra', '99928' : 'georgia',
            '99935' : 'haiti', '99936' : 'bhutan', '99942' : 'armenia', '99943' : 'albania', '99974' : 'bolivia',
            '99975' : 'mongolia', '99989' : 'paraguay'}
    check_list = []
    books['isbn_country'] = 'na'
    for idx in range(len(books)):
        isbn = books['isbn'][idx][:5]
        if isbn[0] in isbn_code.keys():
            books.at[idx, 'isbn_country'] = isbn_code[isbn[0]]
        elif isbn[:2] in isbn_code.keys():
            books.at[idx, 'isbn_country'] = isbn_code[isbn[0:2]]
        elif isbn[:3] in isbn_code.keys():
            books.at[idx, 'isbn_country'] = isbn_code[isbn[0:3]]
        elif isbn[:4] in isbn_code.keys():
            books.at[idx, 'isbn_country'] = isbn_code[isbn[:4]]
        elif isbn[:] in isbn_code.keys():
            books.at[idx, 'isbn_country'] = isbn_code[isbn[:]]
        else:
            check_list.append(isbn)

    books[books['isbn_country'] == 'na']['isbn_country'] = 'english'
    # category_new replace category_high
    books = books.drop(['category_high'], axis = 1)
    # author
    books['book_author'] = books['book_author'].apply(text_preprocessing_func)




    # 여기부터는 원래 있던 것
    ids = pd.concat([train['user_id'], sub['user_id']]).unique()
    isbns = pd.concat([train['isbn'], sub['isbn']]).unique()

    idx2user = {idx:id for idx, id in enumerate(ids)}
    idx2isbn = {idx:isbn for idx, isbn in enumerate(isbns)}

    user2idx = {id:idx for idx, id in idx2user.items()}
    isbn2idx = {isbn:idx for idx, isbn in idx2isbn.items()}

    train['user_id'] = train['user_id'].map(user2idx)
    sub['user_id'] = sub['user_id'].map(user2idx)
    test['user_id'] = test['user_id'].map(user2idx)

    train['isbn'] = train['isbn'].map(isbn2idx)
    sub['isbn'] = sub['isbn'].map(isbn2idx)
    test['isbn'] = test['isbn'].map(isbn2idx)

    field_dims = np.array([len(user2idx), len(isbn2idx)], dtype=np.uint32)

    data = {
            'train':train,
            'test':test.drop(['rating'], axis=1),
            'field_dims':field_dims,
            'users':users,
            'books':books,
            'sub':sub,
            'idx2user':idx2user,
            'idx2isbn':idx2isbn,
            'user2idx':user2idx,
            'isbn2idx':isbn2idx,
            }


    return data

def dl_data_split(args, data):
    X_train, X_valid, y_train, y_valid = train_test_split(
                                                        data['train'].drop(['rating'], axis=1),
                                                        data['train']['rating'],
                                                        test_size=args.TEST_SIZE,
                                                        random_state=args.SEED,
                                                        shuffle=True
                                                        )
    data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
    return data

def dl_data_loader(args, data):
    train_dataset = TensorDataset(torch.LongTensor(data['X_train'].values), torch.LongTensor(data['y_train'].values))
    valid_dataset = TensorDataset(torch.LongTensor(data['X_valid'].values), torch.LongTensor(data['y_valid'].values))
    test_dataset = TensorDataset(torch.LongTensor(data['test'].values))

    train_dataloader = DataLoader(train_dataset, batch_size=args.BATCH_SIZE, shuffle=args.DATA_SHUFFLE)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.BATCH_SIZE, shuffle=args.DATA_SHUFFLE)
    test_dataloader = DataLoader(test_dataset, batch_size=args.BATCH_SIZE, shuffle=False)

    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader

    return data
