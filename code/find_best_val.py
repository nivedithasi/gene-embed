import os
import json
import pandas as pd

def extract_unique_values(df, column_list):
    unique_values = {}
    for column in column_list:
        unique_values[column] = set(df[column].unique())
    return unique_values

pd.set_option("display.max_columns", 30)

# Function to read JSON file
def read_json_file(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def retrieve_test_of_best_val(source_folder):
    dfs = []

    for files_to_gather in ['val_scores.json', 'test_scores.json']:
        table = []
        for folder, dirs, files in os.walk(source_folder):
            folder_name = os.path.basename(folder)
            test_scores_file = os.path.join(folder, files_to_gather)
            config_file = os.path.join(folder, 'config.json')
            
            # Read contents of test_scores.json
            try:
                test_scores_content = read_json_file(test_scores_file)
            except FileNotFoundError:
                test_scores_content = {'error': 'File not found'}
            except json.JSONDecodeError:
                test_scores_content = {'error': 'JSON decoding error'}

            # Read contents of config.json
            try:
                config_content = read_json_file(config_file)
            except FileNotFoundError:
                config_content = {'error': 'File not found'}
            except json.JSONDecodeError:
                config_content = {'error': 'JSON decoding error'}

            table.append({
                'Folder Name': folder_name, **test_scores_content, **config_content
            })
        dfs.append(pd.DataFrame(table))

    df_val, df_test = dfs

    cols = ['Folder Name', 'p@top{k}', 'auprc', 'auroc', 'enrichment@top{k}', 'nhead', 'nhid', 'learning_rate', 'nlayers', 'dropout']

    df_test_slice = df_test[cols]
    df_val_slice = df_val[cols]

    # Sort the dataframe by p@top{k} and auroc in descending order
    sorted_df = df_val_slice.sort_values(by=['p@top{k}', 'auroc'], ascending=False)

    # Get the name of the file with the highest p@top{k} score
    top_file_name = sorted_df.iloc[0]['Folder Name']

    # print(f"The file with the highest p@top{{k}} score is: {top_file_name}")

    # Filter the test dataframe to get the row with the highest score
    highest_score_test_row = df_test[df_test['Folder Name'] == top_file_name]

    print("-----------------------------")
    print("EXPERIMENT TYPE:", source_folder)
    print("-----------------------------")
    print(f"Test scores for the best val model ({top_file_name}):")
    print(f"    p@topK = {round(float(highest_score_test_row['p@top{k}']), 3)}")
    print(f"    enrichment@topk = {round(float(highest_score_test_row['enrichment@top{k}']), 3)}")


if __name__ == '__main__':
    source_folders = ["grid_search_compound",
                      "grid_search_compound_permute",
                      "grid_search_gene",
                      "grid_search_gene_permute",
                      "grid_search_pair",
                      "grid_search_pair_permute"]
    for source in source_folders:
        retrieve_test_of_best_val(source)
        print("\n")