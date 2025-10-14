import numpy as np
import pandas as pd
from sklearn.utils import shuffle

def reindex_score_data(file_path, orginal_data_file, prefix=''):
    score_data_file = "score_test_result.csv";
    score_data_file_reindexed = "score_test_result_reindexed.csv"
    dataset = pd.read_csv(file_path+orginal_data_file);
    # display(dataset);
    
    score_test_data = pd.read_csv(file_path+score_data_file,index_col=0);
    # display(score_test_data);
    
    #Map the feature names with indeces
    #Convert the columns to their numerical representations
    name_index_mapping = {};
    index_name_mapping = {};
    colList = dataset.columns.tolist();
    for index, name in enumerate(colList):
        name_index_mapping[name] = index;
        index_name_mapping[index] = name;


    old_new_index_mapping = {};
    sep_str = "_causes_";
    for idx in score_test_data.index:
        ids = idx.split(sep_str);
        cause_str = prefix+index_name_mapping[int(ids[0])];
        effect_str = prefix+index_name_mapping[int(ids[1])];
        temp_idx_str = cause_str+sep_str+effect_str;
        # print(temp_idx_str);
        old_new_index_mapping[idx] = temp_idx_str;
        
    score_test_data.rename(index=old_new_index_mapping, inplace=True);
    # display(score_test_data);

    score_test_data.to_csv(file_path+score_data_file_reindexed);

def merge_scores(score_file_dict, merged_score_data_path):
    df_list = [];
    for file_path, flag in score_file_dict.items():
        print(file_path);
        score_test_result = pd.read_csv(file_path,index_col=0);
        #Keep only whose label is 1.
        if flag:
            score_test_result = score_test_result[score_test_result['Label']==1];
        if score_test_result.isnull().values.any():
            print("DataFrame contains missing values.")
            rows_without_missing_values = score_test_result.notnull().all(axis=1);
            score_test_result = score_test_result[rows_without_missing_values];
        else:
            print("DataFrame does not contain missing values.")
        df_list.append(score_test_result);

    score_data_df = pd.concat(df_list);
    # display(score_data_df);
    
    score_data_df = shuffle(score_data_df);
    # display(score_data_df);

    score_data_df.to_csv(merged_score_data_path);
    
        
        
    