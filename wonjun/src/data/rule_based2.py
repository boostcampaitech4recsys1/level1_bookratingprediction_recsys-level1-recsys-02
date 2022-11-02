def rule_base(submission: pd.DataFrame, data: dict) -> pd.DataFrame:
    user_ids = data["user2rating"] # train에 있는 애들
    book_isbns = data["isbn2rating"] # train에 있는 책들
    mean_rating = data["train_mean_rating"] # 전체 평균 값
    
    print("-----------------Start Rule Base-----------------")

    submission.loc[~submission['user_id'].isin(user_ids) & ~submission['isbn'].isin(book_isbns), 'rating'] = mean_rating
    submission.loc[~submission['user_id'].isin(user_ids) & submission['isbn'].isin(book_isbns), 'rating'] = submission.loc[~submission['user_id'].isin(user_ids) & submission['isbn'].isin(book_isbns), 'rating'].map(book_isbns)
    submission.loc[submission['user_id'].isin(user_ids) & ~submission['isbn'].isin(book_isbns), 'rating'] = submission.loc[submission['user_id'].isin(user_ids) & ~submission['isbn'].isin(book_isbns), 'rating'].map(user_ids)
    
    return submission
