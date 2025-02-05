import os
import pandas as pd
import logging as log
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from markdown_pdf import MarkdownPdf, Section
from src.utils import *
from src.testing.stages.response_schemas import *
from src.testing.process_inputs import get_user_prompt


def threshold_similarity(
    categories: list[str], unified_categories: list[str], threshold: float = 0.5
    ) -> bool:
    """
    Calculates similarity of category between unified categories definitions. 
    If above cosine similarity threshold, return False.
    """
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


def json_to_pdf(
    test_dir: str, write_dir: str, case: str, stage: str
    ) -> dict[str, str] | None:
    """
    Converts JSON objects to PDFs.
    """
    cases = get_cases(case=case)
    
    text = {i: "" for i in cases}
    stage_1r_data_allias = {i: {} for i in cases}
    for i in cases:
        text[i] = f"# Stage {stage} Categories\n\n"
        instance_types = get_instance_types(case=i)

        stage_1_dirs = [
            os.path.join(write_dir, f"stage_1_{t}")
            for t in instance_types
        ]
        stage_1r_dirs = [
            os.path.join(write_dir, f"stage_1r_{t}")
            for t in instance_types
        ]
        stage_1c_dir = os.path.join(write_dir, "stage_1c")
        stage_1c_parts = [
            os.path.join(stage_1c_dir, f"part_{k}")
            for k in range(1, 3)
        ]

        # Load stage data
        if check_directories(stage_1_dirs):
            stage_1_data = {
                t: load_json(
                    os.path.join(dir, f"stg_1_{t}_response.txt"),
                    Stage1
                ) 
                for t, dir in zip(instance_types, stage_1_dirs)
            }
        if check_directories(stage_1r_dirs):
            try:
                stage_1r_data = {
                    t: load_json(
                        os.path.join(
                            dir, f"stg_1r_{t}_response.txt"
                        ),
                        Stage1r
                    ) 
                    for t, dir in zip(instance_types, stage_1r_dirs)
                }
                stage_1r_data_allias[i] = stage_1r_data
            except ValueError:
                pass
        if check_directories([stage_1c_parts[1]]):
            stage_1c_2_data = load_json(
                os.path.join(
                    stage_1c_parts[1],
                    f"stg_1c_2_response.txt"
                ),
                Stage1r
            )

        if stage in {'1', '1r'}:
            for t in instance_types:
                categories = (
                    stage_1_data[t].categories if stage == '1'
                    else stage_1r_data[t].refined_categories
                )
                text[i] += format_categories(
                    categories,
                    initial_text=f"## {t.capitalize()} Categories\n\n"
                )
    if stage == '1c':
        categories = stage_1c_2_data.refined_categories
        stage_1c_categories = [cat for cat in categories]
        text = f"# Stage 1c Categories\n\n"
        text += format_categories(
            categories,
            initial_text="## Unified Categories\n\n"
        )
        for i in cases:
            instance_types = get_instance_types(case=i)
            text += f"## *{i.capitalize()} Categories*\n\n"
            for t in instance_types:
                stage_1r_categories = [
                    cat  for cat in stage_1r_data_allias[i][t].refined_categories
                ]
                valid_categories = threshold_similarity(
                    categories=stage_1r_categories,
                    unified_categories=stage_1c_categories
                )
                if len(valid_categories) > 0:
                    text += format_categories(
                        valid_categories,
                        initial_text=f"## {t.capitalize()} Categories\n\n"
                    )
    for i in cases:
        pdf = MarkdownPdf(toc_level=1)
        if case == 'uni_switch' and stage not in {'1c'}:
            pdf.add_section(Section(text[i], toc=False))
            pdf.save(os.path.join(
                test_dir,
                f"stg_{stage}_{i}_categories.pdf"
            ))
        else:
            section = text[i] if stage not in {'1c'} else text
            pdf.add_section(Section(section, toc=False))
            pdf.save(os.path.join(
                test_dir,
                f"stg_{stage}_categories.pdf"
            ))
    return None



def write_test(
    write_dir: str, stage: str, instance_type: str, system: str,
    user: str, response: dict, n: int = None, window_number: str = None
    ) -> None:
    """
    Writes the raw prompts & outputs for tests.
    """
    # Raw response & prompts
    if stage in {'1', '1r'}:
        stage_dir = os.path.join(
            write_dir,
            f"stage_{stage}_{instance_type}"
        )
        os.makedirs(stage_dir, exist_ok=True)
        write_file(os.path.join(
            stage_dir, f'stg_{stage}_{instance_type}_sys_prmpt.txt'
        ), str(system))
        write_file(os.path.join(
            stage_dir, f'stg_{stage}_{instance_type}_user_prmpt.txt'
        ), str(user))
        write_file(os.path.join(
            stage_dir, f'stg_{stage}_{instance_type}_response.txt'
        ), str(response))
    
    if stage in {'2', '3'}:
        stage_dir = os.path.join(
            write_dir,
            f"stage_{stage}_{instance_type}"
        )
        response_dir = os.path.join(stage_dir, "responses")
        prompt_dir = os.path.join(stage_dir, "prompts")
        os.makedirs(response_dir, exist_ok=True)
        os.makedirs(prompt_dir, exist_ok=True)
        if n == 1:
            write_file(os.path.join(
                stage_dir,
                f'stg_{stage}_{instance_type}_sys_prmpt.txt'
            ), str(system))
        write_file(os.path.join(
            prompt_dir, f'{window_number}_user_prmpt.txt'
        ), str(user))
        write_file(os.path.join(
            response_dir, f'{window_number}_response.txt'
        ), str(response))
    
    if stage in {'1c'}:
        part = 1 if instance_type == '1c_1' else 2
        stage_dir = os.path.join(write_dir, "stage_1c", f"part_{part}")
        os.makedirs(stage_dir, exist_ok=True)
        write_file(os.path.join(
            stage_dir, f'stg_{stage}_{part}_sys_prmpt.txt'
        ), str(system))
        write_file(os.path.join(
            stage_dir, f'stg_{stage}_{part}_user_prmpt.txt'
        ), str(user))
        write_file(os.path.join(
            stage_dir, f'stg_{stage}_{part}_response.txt'
        ), response)
    

def write_test_info(
    meta: dict, write_dir: str, model_info: object, data_file: dict, stage: str
    ) -> None:
    """
    Writes test info.
    """
    case = data_file['case']
    cases = get_cases(case=case)
    first_case = next(iter(meta))
    first_key = next(iter(meta[first_case]))
    if meta[first_case][first_key] == 0:
        log.warning("No meta data found. Skipping test info.")
        return None
    test_info_lines = [
        "MODEL INFORMATION:",
        "",
        f"Model: {model_info.model}",
        f"Temperature: {model_info.temperature}",
        f"Max-tokens: {model_info.max_completion_tokens}",
        f"Seed: {model_info.seed}",
        f"Top-p: {model_info.top_p}",
        f"Frequency penalty: {model_info.frequency_penalty}",
        f"Presence penalty: {model_info.presence_penalty}",
        "",
        "TEST INFORMATION:",
        "",
        f"Test date/time: {
            datetime.fromtimestamp(
                meta[first_case][first_key].created
            ).strftime('%Y-%m-%d %H:%M:%S')
        }",
        f"Case: {case}",
        f"Summary: {data_file['ra']}",
        f"Treatment: {data_file['treatment']}",
        f"System fingerprint: {
            meta[first_case][first_key].system_fingerprint
        }",
        "",
    ]

    # Loop through each window
    total = {'completion_tokens': 0, 'prompt_tokens': 0, 'total_tokens': 0}
    if case == 'uni_switch' and stage in {'1c'}:
        cases = ['uni_switch']
    for i in cases:
        for key, value in meta[i].items():
            if case == 'uni_switch' and stage not in {'1c'}:
                test_info_lines.append(
                    f"{i.upper()} {key.upper()} PROMPT USAGE:"
                )
            else:
                test_info_lines.append(f"{key.upper()} PROMPT USAGE:")
            test_info_lines.extend([
                "",
                f"Completion tokens: {value.usage.completion_tokens}",
                f"Prompt tokens: {value.usage.prompt_tokens}",
                f"Total tokens: {value.usage.total_tokens}",
                "",
            ])
            total['completion_tokens'] += value.usage.completion_tokens
            total['prompt_tokens'] += value.usage.prompt_tokens
            total['total_tokens'] += value.usage.total_tokens
    
    # Writing totals
    test_info_lines.extend([
        "TOTAL PROMPT USAGE:",
        "",
        f"Completion tokens: {total['completion_tokens']}",
        f"Prompt tokens: {total['prompt_tokens']}",
        f"Total tokens: {total['total_tokens']}",
    ])

    # Define the directory and file path for the test info file
    info_dir = os.path.join(
        write_dir, "_test_info",
        f"stg{stage}_test_info.txt"
    )
    os.makedirs(os.path.dirname(info_dir), exist_ok=True)
    write_file(info_dir, "\n".join(test_info_lines))
    

def build_gpt_output(
    test_dir: str, main_dir: str, write_dir: str, case: str, ra: str,
    treatment: str, stage: str, max_instances: int = None
    ) -> None:
    """
    Builds GPT classification output for stages 2 or 3.
    """
    cases = get_cases(case=case)
    
    for i in cases:
        if stage not in {'2', '3'}:
            raise ValueError(
                f"""Can only build GPT output if stage in ['2', '3']. 
                Got: {stage}"""
            )
        
        instance_types = get_instance_types(case=i)
        test_df = get_user_prompt(
            case=i,
            ra=ra,
            treatment=treatment,
            stage='2',
            main_dir=main_dir,
            write_dir=write_dir,
            max_instances=max_instances
        )[i]
        response_dirs = {
            t: os.path.join(
                write_dir,
                f"stage_{stage}_{t}",
                "responses"
            )
            for t in instance_types
        }
    
        response_list = {t: [] for t in instance_types}
        df_list = []
        for t in instance_types:
            response_dir = response_dirs[t]
            for file in os.listdir(response_dir):
                file_path = os.path.join(response_dir, file)
                
                response_data = {}
                if stage == '3':
                    json_data = load_json(file_path, Stage3)
                    for k in json_data.category_ranking:
                        response_data['reasoning'] = json_data.reasoning
                        response_data[k.category_name] = k.rank
                else:
                    json_data = load_json(file_path, Stage2)
                    for k in json_data.assigned_categories:
                        response_data['reasoning'] = json_data.reasoning
                        response_data[k] = 1
                
                response_data['window_number'] = int(json_data.window_number)
                response_list[t].append(response_data)
            
            df = pd.DataFrame.from_records(response_list[t])
            df = pd.merge(test_df[t], df, on='window_number', how='outer')
            
            common_columns = ['summary_1', 'summary_2',
                              'window_number', 'reasoning']
            distinct_columns = {
                'first': 'cooperation',
                'switch': 'cooperation',
                'uniresp': 'unilateral_other_cooperation',
                'uni': 'unilateral_cooperation'
            }
            
            remove_columns = common_columns + [distinct_columns[i]]
            df_remove_cols = df.columns.intersection(remove_columns)
            df_dropped = df.drop(columns=df_remove_cols)
            category_columns = df_dropped.columns.to_list()
            
            rename_dict = {col: f'{t}_{col}' for col in category_columns}
            df = df.rename(columns=rename_dict)
            df_list.append(df)
        final_df = pd.concat(
            [df_list[0], df_list[1]], ignore_index=True, sort=False
        )
        final_df = final_df.drop(
            columns=final_df.columns.intersection(
                ['summary_1', 'summary_2'] + [distinct_columns[i]]
            )
        )
        final_df = final_df.fillna(0)

        # Merge with raw DataFrame and save
        raw_df = pd.read_csv(os.path.join(
            main_dir, "data", "raw", f"{i}_{treatment}_{ra}.csv"
        ))
        output_df = pd.merge(raw_df, final_df, on='window_number')
        if case == 'uni_switch':
            output_df.to_csv(os.path.join(
                test_dir, f"stg_{stage}_{i}_final_output.csv"
            ), index=False)
        else:
            output_df.to_csv(os.path.join(
                test_dir, f"stg_{stage}_final_output.csv"
            ), index=False)
            

def format_categories(categories: list, initial_text: str = "") -> str:
    """
    Format category text for prompts.
    """
    formatted_text = initial_text
    for category in categories:
        formatted_text += f"### {category.category_name} \n\n"
        formatted_text += f"**Definition**: {category.definition}\n\n"
        try:
            formatted_text += f"**Examples**:\n\n"
            for idx, example in enumerate(category.examples, start=1):
                formatted_text += f"  {idx}. Window number:"
                formatted_text += f"{example.window_number},"
                formatted_text += f"Reasoning: {example.reasoning}\n\n"
        except KeyError:
            pass
    return formatted_text