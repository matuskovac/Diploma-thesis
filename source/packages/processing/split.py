import random
import pandas as pd



def temp(df ,all_usernames, new_user_usernames, y_column):
    unique_patterns_id = df.groupby(y_column)['id'].unique()

    pattern_ids_to_train = []
    pattern_ids_to_val = []
    pattern_ids_to_test = []
    for user, patterns_id in unique_patterns_id.iteritems():
        if(user in new_user_usernames):
            pattern_ids_to_val += list(patterns_id[:round(len(patterns_id)*0.5)])
            pattern_ids_to_test += list(patterns_id[round(len(patterns_id)*0.5):])
        else:        
            pattern_ids_to_train += list(patterns_id[:round(len(patterns_id)*0.7)])
            pattern_ids_to_val += list(patterns_id[round(len(patterns_id)*0.7):round(len(patterns_id)*0.85)])
            pattern_ids_to_test += list(patterns_id[round(len(patterns_id)*0.85):])

    
    mask = [True if x['id'] in pattern_ids_to_train else False for i, x in df.iterrows()]
    df_part_train = df[mask]  
    
    mask = [True if x['id'] in pattern_ids_to_val else False for i, x in df.iterrows()]
    df_part_val = df[mask]  
    
    mask = [True if x['id'] in pattern_ids_to_test else False for i, x in df.iterrows()]
    df_part_test = df[mask] 
    
    return df_part_train, df_part_val, df_part_test

def split_to_train_val_test_raw(df, y_column):
    df_shuffled = df.sample(frac=1, random_state=20).reset_index(drop=True)
    
    all_usernames = df_shuffled[y_column].unique()
    new_user_usernames = all_usernames[round(len(all_usernames)*0.8):]
    
    
    df_raw_train, df_raw_val, df_raw_test = temp(df_shuffled.loc[df_shuffled['scenario'] == 'scenario_show_simple'], all_usernames, new_user_usernames,y_column)
    df_raw_train2, df_raw_val2, df_raw_test2 = temp(df_shuffled.loc[df_shuffled['scenario'] == 'scenario_show_complex'], all_usernames, new_user_usernames, y_column)
    
    df_raw_train = pd.concat([df_raw_train,df_raw_train2])
    df_raw_val = pd.concat([df_raw_val,df_raw_val2])
    df_raw_test = pd.concat([df_raw_test,df_raw_test2])
    
    df_raw_train = df_raw_train.sample(frac=1, random_state=20).reset_index(drop=True)
    df_raw_val = df_raw_val.sample(frac=1, random_state=20).reset_index(drop=True)
    df_raw_test = df_raw_test.sample(frac=1, random_state=20).reset_index(drop=True)
    
    return df_raw_train, df_raw_val, df_raw_test

def adapt_dfs_to_users(df_raw_train, df_raw_val, df_raw_test, users, y_column,  filt = 0):
    
    if(filt==1):
        df_train_filtered=df_raw_train.loc[df_raw_train['scenario'] == 'scenario_show_simple']
        df_val_filtered=df_raw_val.loc[df_raw_val['scenario'] == 'scenario_show_simple']
        df_test_filtered=df_raw_test.loc[df_raw_test['scenario'] == 'scenario_show_simple']
    elif (filt == 2):
        df_train_filtered=df_raw_train.loc[df_raw_train['scenario'] == 'scenario_show_complex']
        df_val_filtered=df_raw_val.loc[df_raw_val['scenario'] == 'scenario_show_complex']
        df_test_filtered=df_raw_test.loc[df_raw_test['scenario'] == 'scenario_show_complex']
    else:
        df_train_filtered=df_raw_train
        df_val_filtered=df_raw_val
        df_test_filtered=df_raw_test
    
    mask = [True if x[y_column] in users else False for i, x in df_train_filtered.iterrows()]
    df_train = df_train_filtered[mask]
    
    
    mask = [True if x[y_column] in users else False for i, x in df_val_filtered.iterrows()]
    df_val = df_val_filtered[mask]
    
    mask = [True if x[y_column] not in users else False for i, x in df_val_filtered.iterrows()]
    df_not_users_val = df_val_filtered[mask]
    
    unique_users_paterns = df_val['id'].unique()
    unique_not_users_paterns = df_not_users_val['id'].unique()[:len(unique_users_paterns)]
    
    mask = [True if x['id'] in unique_not_users_paterns else False for i, x in df_not_users_val.iterrows()]
    df_val = pd.concat([df_val, df_not_users_val[mask]])
    
    
    
    mask = [True if x[y_column] in users else False for i, x in df_test_filtered.iterrows()]
    df_test = df_test_filtered[mask]
    
    mask = [True if x[y_column] not in users else False for i, x in df_test_filtered.iterrows()]
    df_not_users_test = df_test_filtered[mask]
    
    unique_users_paterns = df_test['id'].unique()
    unique_not_users_paterns = df_not_users_test['id'].unique()[:len(unique_users_paterns)]
    
    mask = [True if x['id'] in unique_not_users_paterns else False for i, x in df_not_users_test.iterrows()]
    df_test = pd.concat([df_test, df_not_users_test[mask]])
    
    
    return df_train, df_val, df_test