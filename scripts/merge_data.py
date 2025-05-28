##### Base libs
import pandas as pd
import csv
import chardet

##### Config settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000) 

 

# ------------------- DATASET MERGING functions -------------------



def infer_miss_gender(df, gend_col, proba_col, thr = 0.7):
    # use is_female_proba to infer the missing values for gender
    def _infer_miss_gender(proba_col):
        if proba_col < (1 - thr):
            return "m"
        if proba_col > thr:
            return "f"
    
    # create new_gender feature replacing gender missing values based on a threshold on is_female_proba
    df[gend_col] = df[gend_col].fillna(df[proba_col].apply(_infer_miss_gender))
    


def split_real_data(in_file, out_file1, out_file2):
    # function to split the csv data in two, to make it fit in memory
    # split ratio
    split = 0.5

    # count total rows
    with open(in_file, "r", encoding = "utf-8") as f:
        tot_lines = sum(1 for _ in f) - 1

    print(f"Total lines in {in_file}: {tot_lines}")
    split_point = int(tot_lines * split)

    # read and split comments in two csv
    with open(in_file, "r", encoding = "utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)  # Read header

        with open(out_file1, "w", encoding = "utf-8", newline = "") as f1, open(out_file2, "w", encoding = "utf-8", newline = "") as f2:
            writer1 = csv.writer(f1)
            writer2 = csv.writer(f2)

            # write header to both files
            writer1.writerow(header)
            writer2.writerow(header)

            # track line count and split correctly
            for i, row in enumerate(reader):  
                if i < split_point:
                    writer1.writerow(row)
                else:
                    writer2.writerow(row)

    print(f"Splitting completed: {out_file1} and {out_file2}")



def merge_sim_data(mod, seeds, dir_path, experiment):
    # empty df for merging
    df_merge = pd.DataFrame()
    # iterate over all files in the directory, over moderation type and seed
    for mod_type in mod:
        for s in seeds:
            if experiment == "echo-chambers":
                for tox in ["healthy", "toxic"]:
                    # name of the file
                    fname = f"exp_{mod_type}_{tox}_{s}.csv"
                    print(f"Loading {fname}:")
                    # reading file in a df
                    df = pd.read_csv(f"{dir_path}/{experiment}/{fname}", encoding = "utf-8")
                    print(f"{df.shape}\n")
                    # add a feature to represent the PMI used
                    df["pmi_type"] = mod_type
                    # add a feature to represent the toxicity of the experiment
                    df["tox_type"] = tox
                    # concatenate the dfs iteratively    
                    df_merge = pd.concat([df_merge, df], ignore_index = True)
            else:
                fname = f"exp_{mod_type}_{s}.csv"
                print(f"Loading {fname}:")
                # reading file in a df
                df = pd.read_csv(f"{dir_path}/{experiment}/{fname}", encoding = "utf-8")
                print(f"{df.shape}\n")
                
                # add a feature to represent the PMI used
                df["pmi_type"] = mod_type
            
                # concatenate the dfs iteratively    
                df_merge = pd.concat([df_merge, df], ignore_index = True)
    
    print(f"{experiment} data loaded successfully.\n\n")
    return df_merge



def organize_sim_data(df_orig, loc = 7):    
    # function to have a dataframe with content before mod
    # and all related contents after moderation based on the pmi type
    df = df_orig[df_orig["pmi_type"] == "ofsa"].copy()
    df_neut = df_orig[df_orig["pmi_type"] == "neutral"]
    df_emp = df_orig[df_orig["pmi_type"] == "empathizing"]
    df_pres = df_orig[df_orig["pmi_type"] == "prescriptive"]

    # extract content after each pmi type and reset indices
    a_content_neut = df_neut["a_content"].reset_index(drop = True)
    a_content_emp = df_emp["a_content"].reset_index(drop = True)
    a_content_pres = df_pres["a_content"].reset_index(drop = True)

    # combine in a pmi dataframe
    df_pmi = pd.concat([a_content_neut, a_content_emp, a_content_pres], axis = 1)
    df_pmi.columns = ["a_content_neut", "a_content_emp", "a_content_pres"]
    # rename a_content for ofsa
    df.rename(columns = {"a_content": "a_content_ofsa"}, inplace = True)

    # insert the new columns (from position 17 in this case)
    for i, col in enumerate(df_pmi.columns):
        df.insert(loc = loc + i, column = col, value = df_pmi[col])

    return df 