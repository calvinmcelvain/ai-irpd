import os
import pandas as pd
from itertools import product
from src.utils import get_instance_types


def merge_raw_data(case: str, ra: str, main_dir: str) -> None:
    """
    Function that merges the treatments for summary data. Saves to raw folder.
    """
    treatments = ['noise', 'no_noise']
    ras = ['thi', 'eli']
    
    if ra != 'both':
        dfs = {}
        for treatment in treatments:
            file_path = os.path.join(
                main_dir, "data", "raw", f'{case}_{treatment}_{ra}.csv'
            )
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            dfs[treatment] = pd.read_csv(file_path)
    else:
        all_dfs = {}
        for r, treatment in list(product(ras, treatments)):
            file_path = os.path.join(
                main_dir, "data", "raw", f'{case}_{treatment}_{ra}.csv'
            )
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            all_dfs[f"{treatment}_{r}"] = pd.read_csv(file_path)
        
        dfs = {}
        for treatment in treatments:
            dfs[treatment] = pd.merge(
                all_dfs[f"{treatment}_eli"],
                all_dfs[f"{treatment}_thi"],
                how='outer'
            )
        
        for treatment in treatments:
            dfs[treatment].to_csv(os.path.join(
                main_dir, "data", "raw", f'{case}_{treatment}_both.csv'
            ), index=False)
        
    noise = dfs['noise']
    no_noise = dfs['no_noise']
    merged_df = pd.concat([no_noise, noise], ignore_index=True, sort=False)
    merged_df.to_csv(os.path.join(
        main_dir, "data", "raw", f'{case}_merged_{ra}.csv'
    ), index=False)


def test_dfs(case: str, ra: str, main_dir: str) -> None:
    """
    Function that creates test data. Saves to test folder.
    """
    noise = pd.read_csv(os.path.join(
        main_dir, "data", "raw", f'{case}_noise_{ra}.csv'
    ))
    no_noise = pd.read_csv(os.path.join(
        main_dir, "data", "raw", f'{case}_no_noise_{ra}.csv'
    ))
    merged = pd.read_csv(os.path.join(
        main_dir, "data", "raw", f'{case}_merged_{ra}.csv'
    ))
    df_set = {
        'noise': noise,
        'no_noise': no_noise,
        'merged': merged
    }
    
    keep_columns = {
        1: ['summary_1', 'summary_2', 'window_number',
            'cooperate', 'treatment'],
        2: ['summary_1', 'summary_2', 'window_number',
            'unilateral_other_cooperate', 'unilateral_other_defect',
            'treatment'],
        3: ['summary_1', 'summary_2', 'window_number',
            'unilateral_cooperate', 'unilateral_defect', 'treatment']
    }
    summary_columns = ['summary_1', 'summary_2']
    window_columns = {
        1: {
            'coop': 'cooperate',
            'def': 'defect'
            },
        2: {
            'ucoop': 'unilateral_other_cooperate',
            'udef': 'unilateral_other_defect'
            },
        3: {
            'ucoop': 'unilateral_cooperate',
            'udef': 'unilateral_defect'
            }
    }
    column_map = {
        'first': 1,
        'switch': 1,
        'uniresp': 2,
        'uni': 3
    }
        
    
    keep_columns = keep_columns[column_map[case]]
    window_columns = window_columns[column_map[case]]
    
    instance_types = get_instance_types(case=case)
    
    for treatment, df in df_set.items():
        df = df[df.columns.intersection(keep_columns)]
        if case in {'switch', 'first'}:
            df = df.copy()
            df.loc[:, 'defect'] = df['cooperate'].apply(
                lambda x: 1 if x == 0 else 0
            )
        
        df_column = df.columns.intersection(summary_columns)
        df.loc[:, df_column] = df[df_column].replace(',', '', regex=True)
        
        for instance_type in instance_types:
            instance_df = df.loc[(df[window_columns[instance_type]] == 1)]
            instance_df = instance_df.drop(window_columns.values(), axis=1)
            instance_df.to_csv(os.path.join(
                main_dir, "data", "test",
                f'{case}_{treatment}_{ra}_{instance_type}.csv'
            ), index=False)