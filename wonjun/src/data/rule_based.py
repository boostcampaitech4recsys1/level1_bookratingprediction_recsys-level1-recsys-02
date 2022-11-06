import pandas as pd
from tqdm import tqdm


def rule_base(submission:pd.DataFrame, data:dict) -> pd.DataFrame:
    print("-----------------Start Rule Base-----------------")
    for idx in tqdm(range(len(submission))):
        
        if not submission["user_id"][idx] in data["unique_user_ids"] and not submission["isbn"][idx] in data["unique_isbns"]:
            submission.at[idx, "rating"] = data["train_mean_rating"]
        elif not submission["user_id"][idx] in data["unique_user_ids"]:
            submission.at[idx, "rating"] = data["isbn2rating"][submission["isbn"][idx]]
        elif not submission["isbn"][idx] in data["unique_isbns"]:
            submission.at[idx, "rating"] = data["user2rating"][submission["user_id"][idx]]
        else:
            pass

    return submission