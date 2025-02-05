import os
import pandas as pd
from utils import *
from src.testing.stages.response_schemas import *
from src.testing.process_outputs import threshold_similarity


def check_completed_requests(
    test_dir: str, instance_type: str, stage: str
    ) -> list:
    """
    Returns the requests already made for stage 2 and 3.
    """
    response_dir = os.path.join(
        test_dir, f"stage_{stage}_{instance_type}", "responses"
    )
    if not os.path.exists(response_dir):
        requests = []
    else:
        response_files = sorted(
            [f for f in os.listdir(response_dir) if f.endswith("response.txt")]
        )
        requests = [
            int(re.search(r'(\d+)_response\.txt', file).group(1))
            for file in response_files
        ]
    return requests


def get_system_prompt(
    case: str, ra: str, treatment: str, stage: str,
    prompt_dir: str, write_dir: str
    ) -> dict:
    """
    Gets and returns dictionary of system prompts.
    """
    cases = get_cases(case=case)
    
    system_prompts = {i: {} for i in cases}
    if stage not in {'1c'}:
        for i in cases:
            instance_types = get_instance_types(case=i)

            if stage in {'2', '3'}:
                output = json_to_prompt(
                    case=i,
                    write_dir=write_dir,
                    stage=stage
                )[i]
                markdown_prompt = file_to_string(
                    os.path.join(
                        prompt_dir, i, ra,
                        f"stg_{stage}_{treatment}.md"
                    )
                )
                for t in instance_types:
                    system_prompts[i][t] = f"{markdown_prompt}\n{output[t]}"
            
            if stage in {'0', '1', '1r'}:
                for t in instance_types:
                    if stage == '1':
                        system_prompts[i][t] = file_to_string(
                            os.path.join(
                                prompt_dir, i, ra,
                                f"stg_{stage}_{treatment}_{t}.md"
                            )
                        )
                    else:
                        system_prompts[i][t] = file_to_string(
                            os.path.join(
                                prompt_dir, i, ra,
                                f"stg_{stage}_{treatment}.md"
                            )
                        )
    else:
        system_prompts = {}
        system_prompts['1c_1'] = file_to_string(
            os.path.join(prompt_dir, case, ra, f"stg_1c_{treatment}.md")
        )
        system_prompts['1c_2'] = file_to_string(
            os.path.join(prompt_dir, case, ra, f"stg_1r_{treatment}.md")
        )

    return system_prompts


def get_user_prompt(
    case: str, ra: str, treatment: str, stage: str, main_dir: str,
    write_dir: str, max_instances: int | None = None, **kwargs
    ) -> dict:
    """
    Gets and returns dictionary of user prompts.
    """
    replication = kwargs.get("replication", False)
    cases = get_cases(case=case)
    
    user_prompts = {i: {} for i in cases}
    if stage not in {'1c'}:
        for i in cases:
            instance_types = get_instance_types(case=i)
            data_frames = {
                t: pd.read_csv(
                    os.path.join(
                        main_dir, "data", "test",
                        f"{i}_{treatment}_{ra}_{t}.csv"
                    )
                ).astype({'window_number': int})
                for t in instance_types
            }

            if stage in {'0', '1', '2'}:
                if stage != '1':
                    user_prompts[i] = {
                        t: df[:max_instances] if max_instances else df 
                        for t, df in data_frames.items()
                    }
                else:
                    user_prompts[i] = {
                        t: df.to_dict('records')
                        for t, df in data_frames.items()
                    }
            
            if stage in {'1r'}:
                user_prompts[i] = json_to_prompt(
                    write_dir=write_dir,
                    case=case,
                    stage=stage
                )[i]
            
            if stage in {'3'}:
                user_prompts[i] = {t: [] for t in instance_types}
                for t in instance_types:
                    if max_instances:
                        df = data_frames[t][:max_instances]
                    else:
                        df = data_frames[t]
                    
                    if replication:
                        test_paths = {
                            "noise": os.path.join(
                                main_dir, "output", "uni", "test_34"
                            ),
                            "no_noise": os.path.join(
                                main_dir, "output", "uni", "test_35"
                            ),
                            "merged": os.path.join(
                                main_dir, "output", "uni", "test_36"
                            ),
                        } 
                        response_dir = test_paths[treatment]
                    else:
                        response_dir = os.path.join(
                            write_dir, f"stage_2_{t}/responses"
                        )
                    for file in os.listdir(response_dir):
                        if file.endswith("response.txt"):
                            json_response = load_json(
                                os.path.join(response_dir, file),
                                Stage2
                            )
                            if int(json_response.window_number) in set(
                                df['window_number'].unique()
                                ):
                                summary = df[
                                    df['window_number'] == int(
                                        json_response.window_number
                                    )
                                ].to_dict('records')[0]
                                summary[
                                    'assigned_categories'
                                ] = json_response.assigned_categories
                                user_prompts[i][t].append(summary)    
                    user_prompts[i][t] = pd.DataFrame.from_records(
                        user_prompts[i][t]
                    )
    else:
        part_1_exists = os.path.isdir(os.path.join(
            write_dir, "stage_1c", "part_1"
        ))
        if not part_1_exists:
            if case == 'uni_switch':
                combined_df = pd.read_csv(os.path.join(
                    main_dir, "data", "test",
                    f"{case}_{treatment}_{ra}.csv"
                ))
            else:
                instance_types = get_instance_types(case=case)
                data_frames = {
                    t: pd.read_csv(os.path.join(
                        main_dir, "data", "test",
                        f"{case}_{treatment}_{ra}_{t}.csv"
                    )).astype({'window_number': int})
                    for t in instance_types
                }
                combined_df = pd.concat([
                    df.assign(
                        instance_type=(1 if t == instance_types[0] else 0)
                    ) for t, df in data_frames.items()
                ], ignore_index=True)
            user_prompts = {'1c_1': combined_df.to_dict('records')}
        else:
            user_prompts = json_to_prompt(
                write_dir=write_dir,
                case=case,
                stage=stage
            )

    return user_prompts


def json_to_prompt(
    write_dir: str, case: str, stage: str
    ) -> dict[str, str] | None:
    """
    Gets JSON responses for prompts.
    """
    cases = get_cases(case=case)
    
    output = {i: "" for i in cases}
    stage_1r_data_allias = {i: {} for i in cases}
    for i in cases:
        instance_types = get_instance_types(case=i)
        output[i] = {t: "" for t in instance_types}

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
        if check_directories([stage_1c_parts[0]]):
            stage_1c_1_data = load_json(
                os.path.join(
                    stage_1c_parts[0],
                    f"stg_1c_1_response.txt"
                ),
                Stage1
            )
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
                categories = stage_1_data[t].categories
                output[i][t] = format_categories(categories)
        
        if stage in {'2', '3'}:
            if check_directories(stage_1r_dirs):
                if check_directories(stage_1c_parts):
                    categories = stage_1c_2_data.refined_categories
                    stage_1c_categories = [cat for cat in categories]
                    for t in instance_types:
                        output[i][t] = format_categories(categories)
                        stage_1r_categories = [
                            cat  for cat in stage_1r_data[t].refined_categories
                        ]
                        valid_categories = threshold_similarity(
                            categories=stage_1r_categories,
                            unified_categories=stage_1c_categories
                        )
                        if len(valid_categories) > 0:
                            output[i][t] += format_categories(valid_categories)
                else:
                    for t in instance_types:
                        stage_1r_categories = stage_1r_data[t].categories
                        output[i][t] += format_categories(stage_1r_categories)
    
    if stage == '1c':
        categories = stage_1c_1_data.categories
        output = {}
        output = {'1c_2': format_categories(categories)}
    return output