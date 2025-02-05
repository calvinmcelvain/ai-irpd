# Packages
import sys, os
import numpy as np
import pandas as pd
import re
from dotenv import load_dotenv
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Appending src dir. for module import
sys.path.append(os.path.dirname(os.getcwd()))

# Loading Environment Variables
load_dotenv(os.path.join(os.path.dirname(os.getcwd()), "configs/configs.env"))

# Modules
from src.testing.process_inputs import load_json
from src.utils import get_instance_types
from src.testing.schemas import Stage1, Stage1r


def get_responses(
    test_number: int, stage: str, case: str = 'uni', replication_dir: str = None
    ):
    """
    Gets all responses from replication test.
    """
    if replication_dir is None:
        replication_dir = os.path.join(
            os.getenv('PROJECT_DIRECTORY'), "output", "replication_tests"
        )
    if case == 'uni_switch':
        instance_types = ['ucoop', 'udef', 'coop', 'def']
    else:
        instance_types = get_instance_types(case=case)
    test_dir = os.path.join(replication_dir, f'test_{test_number}')
    
    stage_schemas = {
        '1': Stage1,
        '1r': Stage1r
    }
    
    if stage in {'1', '1r'}:
        schema = stage_schemas[stage]
        responses = {
            t: [
                load_json(os.path.join(
                    test_dir, replication, f"stage_{stage}_{t}",
                    f"stg_{stage}_{t}_response.txt"
                ), schema)
                for replication in os.listdir(test_dir)
                if replication.startswith('replication')
            ]
            for t in instance_types
        }
    elif stage == '1c':
        responses = {
            '1c_1': [],
            '1c_2': []
        }
        for t in instance_types:
            responses[t] = []
        for replication in os.listdir(test_dir):
            if replication.startswith('replication'):
                responses['1c_1'].append(
                    load_json(
                        os.path.join(
                            test_dir, replication, "stage_1c",
                            "part_1", "stg_1c_1_response.txt"
                        ), 
                        Stage1
                    )
                )
                responses['1c_2'].append(
                    load_json(
                        os.path.join(
                            test_dir, replication, "stage_1c",
                            "part_2", "stg_1c_2_response.txt"
                        ), 
                        Stage1r
                    )
                )
                for t in instance_types:
                    responses[t].append(
                        load_json(os.path.join(
                            test_dir, replication, f"stage_1r_{t}",
                            f"stg_1r_{t}_response.txt"
                        ), Stage1r)
                    )
    elif stage in {'2', '3'}:
        final_outputs_dir = os.path.join(test_dir, "final_outputs")
        responses = [
            pd.read_csv(os.path.join(final_outputs_dir, file))
            for file in os.listdir(final_outputs_dir) if file.endswith('.csv')
        ]
    return responses


def name_similarity(test_responses: dict) -> pd.DataFrame:
    """
    Calculating category name similarities.
    """
    dfs = []
    for test in range(len(test_responses)):
        responses = test_responses[test]
        new_values = {key: [] for key in responses.keys()}
        for key, items in responses.items():
            for item in items:
                try:
                    category_names = [
                        category.category_name.replace("_", " ")
                        for category in item.categories
                    ]
                except AttributeError:
                    category_names = [
                        category.category_name.replace("_", " ")
                        for category in item.refined_categories
                    ]
                new_values[key].append(" ".join(category_names))
        
        vectorizer = TfidfVectorizer()
        data = []
        for key, combined_texts in new_values.items():
            if len(combined_texts) < 2:
                continue

            tfidf_matrix = vectorizer.fit_transform(combined_texts)
            sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
            
            upper_triangle = np.triu_indices_from(sim_matrix, k=1)
            upper_triangle_values = sim_matrix[upper_triangle]
            
            for sim in upper_triangle_values:
                data.append({
                    'instance_type': key,
                    'similarity': sim,
                    'test': test
                })
        
        dfs.append(pd.DataFrame(data))
    
    return pd.concat(dfs, axis=0, ignore_index=True)
    

def definition_similarity(test_responses: dict) -> pd.DataFrame:
    """
    Calculates category definition similarity by matching pairs of definitions 
    with the maximum cosine similarity, with respect to the matching scheme.
    """
    dfs = []
    for test in range(len(test_responses)):
        responses = test_responses[test]
        data = []
        for key in responses.keys():
            definitions_with_ids = []
            for replication_id, replication in enumerate(responses[key]):
                try:
                    for category in replication.categories:
                        definitions_with_ids.append(
                            (replication_id, category.definition)
                        )
                except AttributeError:
                    for category in replication.refined_categories:
                        definitions_with_ids.append(
                            (replication_id, category.definition)
                        )
        
            definitions = [item[1] for item in definitions_with_ids]
            replication_ids = [item[0] for item in definitions_with_ids]
            
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(definitions)
            similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
            
            # EX: (sim_score, rep_id1, rep_id2)
            similarity_scores = []
            for i, j in combinations(range(len(definitions)), 2):
                if replication_ids[i] != replication_ids[j]:
                    similarity_scores.append((similarity_matrix[i, j], i, j))
            
            similarity_scores.sort(reverse=True, key=lambda x: x[0])
            
            matched = set()
            for sim, i, j in similarity_scores:
                if i not in matched and j not in matched:
                    matched.add(i)
                    matched.add(j)
                    data.append({
                        'instance_type': key,
                        'similarity': sim,
                        'test': test
                    })
        dfs.append(pd.DataFrame(data))
    return pd.concat(dfs, axis=0, ignore_index=True)


def unique_categories(test_responses: dict) -> pd.DataFrame:
    """
    Gets the unique categories and counts from all replicates within a 
    replication test
    """
    data = []
    
    for test_idx, responses in enumerate(test_responses):
        category_counts = {}
        for instance_type, items in responses.items():
            for item in items:
                try:
                    categories = [cat.category_name for cat in item.categories]
                except AttributeError:
                    categories = [
                        cat.category_name for cat in item.refined_categories
                    ]
                
                for category in categories:
                    key = (instance_type, category)
                    category_counts[key] = category_counts.get(key, 0) + 1
        
        data.extend({
            'instance_type': instance_type,
            'category': category,
            'count': count,
            'test': test_idx
        } for (instance_type, category), count in category_counts.items())
    
    return pd.DataFrame(data)


def unified_category_similarity(test_responses: dict) -> pd.DataFrame:
    """
    Calculates similarity of categories and unified categories definitions.
    """
    data = []
    replication_len = len(test_responses['1c_2'])
    vectorizer = TfidfVectorizer()
    for test in range(replication_len):
        unified_cats = test_responses['1c_2'][test].refined_categories
        for t in test_responses.keys():
            if t in {'ucoop', 'udef', 'coop', 'def'}:
                other_cats = test_responses[t][test].refined_categories
                num_cats = len(other_cats)
                all_cat_names = [
                    cat.category_name.replace("_", " ")
                    for cat in unified_cats + other_cats
                ]
                all_cat_defs = [
                    cat.definition for cat in unified_cats + other_cats
                ]
                name_mat = vectorizer.fit_transform(all_cat_names)
                def_mat = vectorizer.fit_transform(all_cat_defs)
                name_sim = cosine_similarity(
                    name_mat[:num_cats], name_mat[num_cats:]
                )
                def_sim = cosine_similarity(
                    def_mat[:num_cats], def_mat[num_cats:]
                )
                df = pd.DataFrame({
                    "name_sim": pd.DataFrame(name_sim).max().to_numpy(),
                    "def_sim": pd.DataFrame(def_sim).max().to_numpy(),
                    "type": t,
                })
                data.append(df)
    return pd.concat(data, axis=0, ignore_index=True)
    


def categorizations(responses: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Gets categorizations of Stage 2 or Stage 3 and returns a DataFrame of
    mean consistency.
    """
    instance_types = ['ucoop', 'udef']
    category_names = [
        i for i in responses[0].columns
        if i.startswith(instance_types[0])
        or i.startswith(instance_types[1])
    ] + ['window_number']
    
    trimmed_responses = [
        response[response.columns.intersection(category_names)]
        for response in responses
    ]
    merged_responses = pd.concat(trimmed_responses)
    merged_responses = merged_responses.groupby('window_number').var()
    merged_responses = merged_responses.reset_index()
    merged_responses = merged_responses.drop(['window_number'], axis=1)
    
    data = []
    for instance_type in instance_types:
        mean_consistency = 1 - merged_responses.filter(like=instance_type)
        mean_consistency = mean_consistency.mean(axis=1)
        for value in mean_consistency:
            data.append({
                'instance_type': instance_type,
                'mean_consistency': value
            })
    
    return pd.DataFrame(data)
