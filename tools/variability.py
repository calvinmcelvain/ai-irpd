import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict
from itertools import combinations
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
sys.path.append(Path().resolve().parent.as_posix())
sys.path.append((Path().resolve().parent / "src").as_posix())
from src.output_manager import OutputManager
from src.utils import validate_json_string
from src.testing.stages.schemas import *
from src.get import Get


class Variability:
    SCHEMA_MAP = {
        "1": Stage1Schema,
        "1r": Stage1rSchema,
        "1c": {
            "part_1": Stage1Schema,
            "part_2": Stage1rSchema,
        },
        "2": Stage2Schema,
        "3": Stage3Schema
    }
    
    def __init__(self, responses: OutputManager):
        self.responses = responses
    
    @staticmethod
    def _get_category_att(output):
        if hasattr(output, "categories"):
            return output.categories
        if hasattr(output, "refined_categories"):
            return output.refined_categories
        if hasattr(output, "assigned_categories"):
            return output.assigned_categories
        if hasattr(output, "category_ranking"):
            return output.category_ranking
    
    @staticmethod
    def _jaccard_sim(s1: set, s2: set):
        return len(s1.intersection(s2)) / len(s1.union(s2))
    
    def _threshold_similarity(self, categories: List, unified_categories: List, threshold: float):
        all_cat_names = [
            cat.category_name.replace("_", " ")
            for cat in categories + unified_categories
        ]
        all_cat_defs = [
            cat.definition for cat in categories + unified_categories
        ]
        cat_names_w_ids = list(enumerate(all_cat_names[:len(categories)]))
        
        vectorizer = TfidfVectorizer()
        tfidf_matrix_names = vectorizer.fit_transform(all_cat_names)
        tfidf_matrix_defs = vectorizer.fit_transform(all_cat_defs)
        
        num_categories = len(cat_names_w_ids)
        
        sim_matrix_names = cosine_similarity(
            tfidf_matrix_names[:num_categories], 
            tfidf_matrix_names[num_categories:]
        )
        sim_matrix_defs = cosine_similarity(
            tfidf_matrix_defs[:num_categories], 
            tfidf_matrix_defs[num_categories:]
        )
        sim_matrix = sim_matrix_names + sim_matrix_defs
        
        results = []
        for cat_idx, sim_array in enumerate(sim_matrix):
            if all(sim_array < threshold):
                results.append(categories[cat_idx])
        return results

    def category_sims(self, threshold: float):
        data = []
        stages = {"1", "1r", "1c"}

        for test, llm_dict in self.responses.test_runs.items():
            for llm, n_dict in llm_dict.items():
                names = {}
                defs = {}

                for n, _ in n_dict.items():
                    for stage in stages:
                        stage_response = self.responses.get(test, n, llm, stage)
                        
                        if not stage_response:
                            continue
                        
                        schema = self.SCHEMA_MAP[stage]
                        if stage not in names:
                            names[stage] = {}
                            defs[stage] = {}

                        for case, values in stage_response.items():
                            if case not in names[stage]:
                                names[stage][case] = {}
                                defs[stage][case] = {}

                            for instance, instance_value in values.items():
                                if stage == "1c":
                                    schema = self.SCHEMA_MAP[stage][instance]
                                response = validate_json_string(instance_value[0].response, schema)
                                
                                if stage == "1r":
                                    stage_1c = self.responses.get(test, n, llm, "1c", "combined", "part_2")
                                    if stage_1c:
                                        u_response = validate_json_string(stage_1c[0].response, schema)
                                        response.refined_categories = self._threshold_similarity(
                                            response.refined_categories, u_response.refined_categories, threshold
                                        )
                                
                                if instance not in names[stage][case]:
                                    names[stage][case][instance] = []
                                    defs[stage][case][instance] = []

                                for category in self._get_category_att(response):
                                    names[stage][case][instance].append((n, category.category_name.replace("_", " ")))
                                    defs[stage][case][instance].append((n, category.definition))
                
                for stage in names.keys():
                    for c in names[stage].keys():
                        for inst in names[stage][c].keys():
                            replicates = [item[0] for item in names[stage][c][inst]]
                            name_list = [item[1] for item in names[stage][c][inst]]
                            def_list = [item[1] for item in defs[stage][c][inst]]
                            
                            for k, values in {"name": name_list, "definition": def_list}.items():
                                vec = TfidfVectorizer().fit_transform(values)
                                cosine = cosine_similarity(vec, vec)

                                cosine_scores = [
                                    (cosine[i, j], i, j)
                                    for i, j in combinations(range(len(values)), 2)
                                    if replicates[i] != replicates[j]
                                ]
                                jaccard_scores = [
                                    (self._jaccard_sim(set(values[i]), set(values[j])), i, j)
                                    for i, j in combinations(range(len(values)), 2)
                                    if replicates[i] != replicates[j]
                                ]

                                cosine_scores.sort(reverse=True, key=lambda x: x[0])
                                jaccard_scores.sort(reverse=True, key=lambda x: x[0])
                                
                                for method, scores in {"cosine": cosine_scores, "jaccard": jaccard_scores}.items():
                                    matched = set()
                                    for sim, i, j in scores:
                                        if i not in matched and j not in matched:
                                            matched.add(i)
                                            matched.add(j)
                                            data.append({
                                                "test": test,
                                                "llm": llm,
                                                "stage": stage,
                                                "case": case,
                                                "instance": inst,
                                                "type": k,
                                                "threshold": threshold,
                                                "method": method,
                                                "sim": sim
                                            })
        return pd.DataFrame(data)
    
    def unique_cats(self, threshold: float):
        data = []
        stages = {"1", "1r", "1c"}

        for test, llm_dict in self.responses.test_runs.items():
            for llm, n_dict in llm_dict.items():
                names = {}

                for n, _ in n_dict.items():
                    for stage in stages:
                        stage_response = self.responses.get(test, n, llm, stage)
                        
                        if not stage_response:
                            continue
                        
                        schema = self.SCHEMA_MAP[stage]
                        if stage not in names:
                            names[stage] = {}

                        for case, values in stage_response.items():
                            if case not in names[stage]:
                                names[stage][case] = {}

                            for instance, instance_value in values.items():
                                if stage == "1c":
                                    schema = self.SCHEMA_MAP[stage][instance]
                                response = validate_json_string(instance_value[0].response, schema)
                                
                                if stage == "1r":
                                    stage_1c = self.responses.get(test, n, llm, "1c", "combined", "part_2")
                                    if stage_1c:
                                        u_response = validate_json_string(stage_1c[0].response, schema)
                                        response.refined_categories = self._threshold_similarity(
                                            response.refined_categories, u_response.refined_categories, threshold
                                        )
                                
                                if instance not in names[stage][case]:
                                    names[stage][case][instance] = []

                                for category in self._get_category_att(response):
                                    names[stage][case][instance].append(category.category_name)
            for stage in names.keys():
                for c in names[stage].keys():
                    for inst in names[stage][c].keys():
                        cat_names = names[stage][c][inst]
                        unique = set(cat_names)
                        for cat in unique:
                            data.append({
                                "test": test,
                                "llm": llm,
                                "stage": stage,
                                "case": case,
                                "instance": inst,
                                "threshold": threshold,
                                "category": cat,
                                "count": len([c for c in cat_names if c == cat])
                            })
        return pd.DataFrame(data)
    


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
