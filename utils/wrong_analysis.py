FILE_1 = 'wrong_analyze.csv'
FILE_2 = 'anothoer_wrong_analyze.csv'

SETUP_NAME = 'test'
OUTPUT_DIR = './'

import pandas as pd
import os.path as osp
import pickle

if __name__ == '__main__':
    # read from csv with header 'Per, Acc, True Class, ...' and extract the first three columns
    # preprocess the files since the columns are not the same, so just extract the first three columns
    file1_lines = []
    file2_lines = []
    with open(FILE_1, 'r') as f:
        for line in f.readlines():
            file1_lines.append(line.split(',')[:3])
    with open(FILE_2, 'r') as f:
        for line in f.readlines():
            file2_lines.append(line.split(',')[:3])
    # remove spaces in the first row
    file1_lines[0] = [x.strip() for x in file1_lines[0]]
    file2_lines[0] = [x.strip() for x in file2_lines[0]]

    df_1 = pd.DataFrame(file1_lines[1:], columns=file1_lines[0])
    df_2 = pd.DataFrame(file2_lines[1:], columns=file2_lines[0])
    # sort by True Class
    df_1 = df_1.sort_values(by=['True Class'])
    df_2 = df_2.sort_values(by=['True Class'])
    
    # column 'Per' and 'Acc' are percentage strings, convert them to float
    df_1['Per'] = df_1['Per'].apply(lambda x: float(x[:-1]))
    df_1['Acc'] = df_1['Acc'].apply(lambda x: float(x[:-1]))
    df_2['Per'] = df_2['Per'].apply(lambda x: float(x[:-1]))
    df_2['Acc'] = df_2['Acc'].apply(lambda x: float(x[:-1]))

    # iterate over the rows of the two dataframes and compare the True Class
    # if they are the same, then calculate the Acc difference and the Per difference, store it into a new csv
    # using column names 'True Class, Acc Diff, Per Diff'
    df_1 = df_1.reset_index(drop=True)
    df_2 = df_2.reset_index(drop=True)
    df_1['Acc Diff'] = df_1['Acc'] - df_2['Acc']
    df_1['Per Diff'] = df_1['Per'] - df_2['Per']
    df_1 = df_1[['True Class', 'Acc Diff', 'Per Diff']]

    # Add df_2's Acc and Per to df_1
    df_1['Acc'] = df_2['Acc']
    df_1['Per'] = df_2['Per']
    
    # sort by Acc Diff
    df_1 = df_1.sort_values(by=['Acc Diff'], ascending=False)
    # save the new csv
    df_1.to_csv(osp.join(OUTPUT_DIR, SETUP_NAME + ".csv"), index=False)

    # iterate through the new csv and transform it into two dict
    # one for Acc Diff, one for Per Diff
    # the key is the True Class, the value is the difference
    acc_diff_dict = {}
    per_diff_dict = {}
    for index, row in df_1.iterrows():
        acc_diff_dict[row['True Class']] = row['Acc Diff']
        per_diff_dict[row['True Class']] = row['Per Diff']
    
    # save the two dict into two pickle files
    with open(osp.join(OUTPUT_DIR, SETUP_NAME + "_acc_diff.pkl"), "wb") as f:
        pickle.dump(acc_diff_dict, f)
    with open(osp.join(OUTPUT_DIR, SETUP_NAME + "_per_diff.pkl"), "wb") as f:
        pickle.dump(per_diff_dict, f)
    