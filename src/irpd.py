import re
import logging
from pathlib import Path
from itertools import product
from stage_outputs import StageOutputs
from utils import get_env_var
from models import LLMModel
from test_config import TestConfig
from logger import setup_logger

setup_logger(clear_logs=True)
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
    

class IRPD:
    _VALID_STAGES = ['0', '1', '1r', '1c', '2', '3']
    _VALID_CASES = ['uni', 'uniresp', 'switch', 'first', 'uni_switch']
    _VALID_RAS = ['thi', 'eli', 'both', 'exp']
    _VALID_TEST_TYPES = ["test", "subtest", "replication", "cross_model_validation"]
    _VALID_TREATMENTS = ['noise', 'no_noise', 'merged']
    _VALID_LLMS = LLMModel._member_names_
    _VALID_LLM_CONFIGS = ["base", "res1", "res2", "res3"]
    REQUEST_TIMEOUT = 60
    MAX_RETRIES = 5

    def __init__(
        self, 
        case: str,
        ras: list[str],
        treatments: list[str],
        stages: list[str],
        test_type: str = "test",
        llms: list[str] = ["GPT_4O_1120"],
        llm_config: str = "base",
        N: int = 1,
        max_instances: int = None,
        project_path: str = None,
        print_response: bool = False,
        new_test: bool = True
    ):
        self.case = self._validate_arg(
            [case], self._VALID_CASES, "cases")[0]
        self.ras = self._validate_arg(
            ras, self._VALID_RAS, "ras")
        self.treatments = self._validate_arg(
            treatments, self._VALID_TREATMENTS, "treatments")
        self.stages = self._validate_arg(
            stages, self._VALID_STAGES, "stages")
        self.test_type = self._validate_arg(
            [test_type], self._VALID_TEST_TYPES, "test_type")[0]
        self.llms = self._validate_arg(
            llms, self._VALID_LLMS, "llms")
        self.llm_config = self._validate_arg(
            [llm_config], self._VALID_LLM_CONFIGS, "llm_config")
        self.N = N
        self.max_instances = max_instances
        self.path = Path(
            project_path if project_path else get_env_var("PROJECT_DIRECTORY")
        )
        self.path_out = self.path / "output"
        self.print_response = print_response
        self.new_test = new_test
        self.stage_outputs = StageOutputs()
        self.product_rlt = product(self.ras, self.treatments, self.llms)
        self.test_configs = []

    def _validate_arg(self, arg: list[str], valid_values: list[str], name: str):
        if not isinstance(arg, list) or not all(isinstance(item, str) for item in arg):
            raise ValueError(f"{name} must be a list of strings.")

        valid_set = set(valid_values)
        index_map = {value: i for i, value in enumerate(valid_values)} 

        valid_items, invalid_items = [], []
        for item in arg:
            (valid_items if item in valid_set else invalid_items).append(item)

        if not valid_items:
            log.error(
                f"All provided `{name}` values are invalid. No valid items remain. "
                f"Allowed values: {valid_values}"
            )
            raise ValueError
        
        if invalid_items:
            log.warning(
                f"Some `{name}` values are invalid and were ignored: {invalid_items}. "
                f"Allowed values: {valid_values}"
            )

        return sorted(valid_items, key=lambda x: index_map[x])
    
    def _generate_model_instance(self, llm: str):
        return getattr(LLMModel, llm).get_model_instance(
                config=self.llm_config, print_response=self.print_response
        )
    
    @staticmethod
    def _get_max_test_number(directory: Path, prefix: str):
        pattern = re.compile(rf"{re.escape(prefix)}(\d+)")
        return max(
            map(int, (match.group(1) for p in directory.iterdir() if (match := pattern.match(p.name)))),
            default=0
        )

    def _generate_test_path(self):
        initializing_stage = any(stage in {"0", "1"} for stage in self.stages)
        if self.test_type == "cross_model_validation":
            self.new_test = (len(self.test_configs) % len(self.llms)) == 0
        else:
            if not self.new_test:
                if len(list(self.product_rlt)) > 1:
                    self.new_test = True
                    log.warning(
                        f"`new_test` must be True if multiple tests are specified."
                        " `new_test` defaulted to True."
                    )
                if self.test_type in {"test", "subtest"} and not initializing_stage:
                    log.error(f"Stages must contain '0' or '1' for {self.test_type}.")
                    raise ValueError("Invalid stage for test type")

        test_dirs = {
            "test": (self.path_out / self.case, "test_"),
            "subtest": (self.path_out / "_subtests", ""),
            "replication": (self.path_out / "replication_tests", "test_"),
            "cross_model_validation": (self.path_out / "cross_model_validation", "test_"),
        }

        base_dir, prefix = test_dirs[self.test_type]
        base_dir.mkdir(parents=True, exist_ok=True)

        test_num = self._get_max_test_number(base_dir, prefix) + (1 if self.new_test else 0)

        return base_dir / (f"{prefix}{test_num}" if prefix else str(test_num))
    
    @staticmethod
    def _generate_output_path(test_config: TestConfig, n: int):
        if test_config.test_type in {"test", "subtest"}:
            outpath = test_config.test_path / "raw"
        elif test_config.test_path in {"replication"}:
            outpath = test_config.test_path / f"replication_{n}"
        elif test_config.test_type in {"cross_model_validation"}:
            outpath = test_config.test_path / test_config.llm / f"replicaiton_{n}"
        return outpath
        
    def _generate_test_config(self, ra: str, treatment: str, llm: str):
        log.info(f"{llm}")
        configs = TestConfig(
            case=self.case,
            ra=ra,
            treatment=treatment,
            llm=llm,
            llm_instance=self._generate_model_instance(llm),
            stages=self.stages,
            test_type=self.test_type,
            test_path=self._generate_test_path(),
            project_path=self.path,
            print_response=self.print_response,
            max_instances=self.max_instances
        )
        self.test_configs.append(configs)
        return configs
    
    def run(self):
        for ra, treatment, llm in self.product_rlt:
            test_config = self._generate_test_config(ra=ra, treatment=treatment, llm=llm)
            
            test_config.test_path.mkdir(exist_ok=True)
            log.info(
                f"Making test directory: {test_config.test_path.relative_to(self.path)}, "
                f"Created: {test_config.test_path.exists()}"
            )
            
            for n, stage in product(range(1, self.N + 1), self.stages):
                outpath = self._generate_output_path(test_config, n)
                outpath.mkdir(exist_ok=True, parents=True)
                log.info(
                    f"Making output directory: {outpath.relative_to(test_config.test_path)}, "
                    f"Created: {outpath.exists()}"
                )
                
                stage_class = globals().get(f"Stage{stage}")
                if stage_class:
                    stage_instance = stage_class(test_config, outpath)
                    log.info(f"Storing output: Stage {stage}")
                    self.stage_outputs.store(f"stage_{stage}_rep_{n}", stage_instance.run())
                    log.info(
                        f"Stage {stage} stored: {self.stage_outputs.has(f"stage_{stage}_rep_{n}")}"
                    )
                else:
                    log.warning(f"Stage {stage} not found.")

    #def _llm_request(self, stage: str, user_input, system_input):
    #    self.llm.configs.max_completion_tokens = (
    #        2000 if stage.startswith('1') else 600
    #    )
    #    output_structure = self._valid_structures[stage]
    #    response_out = self.llm.request(
    #        sys=str(system_input),
    #        user=str(user_input),
    #        output_structure=output_structure,
    #        max_retries=self.MAX_RETRIES,
    #        timeout=self.REQUEST_TIMEOUT,
    #        down_time=self.DOWN_TIME
    #    )
    #    return response_out
    #
    #def _stage_1(self, user: dict, system: dict, write_dir: str):
    #    """
    #    Stage 1 of IRPD test.
    #    """
    #    # Iterating through cases
    #    meta = {t: 0 for t in system.keys()}
    #    for t in system.keys():
    #        json_out = False
    #        retries = 0
    #        while not json_out:
    #            if self.DIFFICULT_LLM:
    #                system[t] = self._difficult_llms_str(system[t])
    #            # LLM requests
    #            response_out = self._llm_request(
    #                stage='1',
    #                user_input=user[t],
    #                system_input=system[t]
    #            )
    #            json_out = self._validate_json_output(
    #                response=response_out.response,
    #                stage='1'
    #            )
    #            if not json_out:
    #                if retries == 3:
    #                    raise TimeoutError("Retries exhuasted")
    #                print(
    #                    f"Output in stage 1 was not a valid JSON. Retrying..."
    #                )
    #                retries += 1
    #        meta[t] = response_out.meta
    #        
    #        # Writing test info
    #        if t.startswith('1c'):
    #            write_test(
    #                write_dir=write_dir,
    #                stage='1c',
    #                instance_type=t,
    #                system=system[t],
    #                user=user[t],
    #                response=response_out.response,
    #            )
    #        else:
    #            write_test(
    #                write_dir=write_dir,
    #                stage='1',
    #                instance_type=t,
    #                system=system[t],
    #                user=user[t],
    #                response=response_out.response,
    #            )
    #    return meta
    #
    #def _stage_1r(self, user: dict, system: dict, write_dir: str):
    #    """
    #    Stage 1r of IRPD test.
    #    """        
    #    # Iterating through cases
    #    meta = {t: 0 for t in system.keys()}
    #    for t in system.keys():
    #        json_out = False
    #        retries = 0
    #        while not json_out:
    #            if self.DIFFICULT_LLM:
    #                system[t] = self._difficult_llms_str(system[t])
    #            
    #            # LLM requests
    #            response_out = self._llm_request(
    #                stage='1r',
    #                user_input=user[t],
    #                system_input=system[t]
    #            )
    #            json_out = self._validate_json_output(
    #                response=response_out.response,
    #                stage='1r'
    #            )
    #            if not json_out:
    #                if retries == 3:
    #                    raise TimeoutError("Retries exhuasted")
    #                print(
    #                    f"Output in stage 1r was not a valid JSON. Retrying..."
    #                )
    #                retries += 1
    #        meta[t] = response_out.meta
    #        
    #        # Writing test info
    #        if t.startswith('1c'):
    #            write_test(
    #                write_dir=write_dir,
    #                stage='1c',
    #                instance_type=t,
    #                system=system[t],
    #                user=user[t],
    #                response=response_out.response,
    #            )
    #        else:
    #            write_test(
    #                write_dir=write_dir,
    #                stage='1r',
    #                instance_type=t,
    #                system=system[t],
    #                user=user[t],
    #                response=response_out.response,
    #            )
    #    return meta
    #
    #def _stage_1c(
    #    self, write_dir: str, user: dict, system: dict, test_info: dict
    #    ):
    #    """
    #    Stage 2 of IRPD test.
    #    """
    #    # Part 1
    #    part_1_meta = self._stage_1(user, {'1c_1': system['1c_1']}, write_dir)
    #
    #    # Part 2
    #    part_2_user = get_user_prompt(
    #        **test_info,
    #        main_dir=self.PATH,
    #        write_dir=write_dir
    #    )
    #    part_2_meta = self._stage_1r(
    #        part_2_user, {'1c_2': system['1c_2']}, write_dir
    #    )
    #    meta_dict = {'1c_1': part_1_meta['1c_1'], '1c_2': part_2_meta['1c_2']}
    #    return meta_dict
    #    
    #
    #def _stage_2(self, write_dir: str, user: dict, system: dict):
    #    """
    #    Stage 2 of IRPD test.
    #    """
    #    # Iterating through cases
    #    meta = {t: 0 for t in system.keys()}
    #    for t in system.keys():
    #        # Checking for requests already made & adjusting user prompt
    #        user_df = user[t]
    #        past_requests = check_completed_requests(
    #            test_dir=write_dir, 
    #            instance_type=t,
    #            stage='2'
    #        )
    #        user_df = user_df[~user_df['window_number'].isin(past_requests)]
    #        if len(user_df) == 0:
    #            break
    #        
    #        # Iterative LLM requests
    #        if self.DIFFICULT_LLM:
    #            system[t] = self._difficult_llms_str(system[t])
    #        system_prompt = system[t]
    #        for i, row in enumerate(user_df.to_dict(orient='records'), start=1):
    #            json_out = False
    #            retries = 0
    #            while not json_out:
    #                # LLM requests
    #                response_out = self._llm_request(
    #                    stage='2',
    #                    user_input=row,
    #                    system_input=system_prompt
    #                )
    #                json_out = self._validate_json_output(
    #                    response=response_out.response,
    #                    stage='2'
    #                )
    #                if not json_out:
    #                    if retries == 3:
    #                        raise TimeoutError("Retries exhuasted")
    #                    print(
    #                        f"Output in stage 2 was not a valid JSON."
    #                        " Retrying..."
    #                    )
    #                    retries += 1
    #            
    #            # Writing request info
    #            write_test(
    #                write_dir=write_dir,
    #                stage='2',
    #                instance_type=t,
    #                system=str(system_prompt),
    #                user=row,
    #                window_number=row['window_number'] 
    #                if isinstance(row, dict) else row.window_number,
    #                response=response_out.response,
    #                n=i
    #            )
    #            
    #            response_meta = response_out.meta
    #            if i == 1:
    #                meta[t] = response_meta
    #            else:
    #                token_usage = response_meta.usage
    #                meta[t].usage.completion_tokens += (
    #                    token_usage.completion_tokens
    #                )
    #                meta[t].usage.prompt_tokens += (
    #                    token_usage.prompt_tokens
    #                )
    #                meta[t].usage.total_tokens += (
    #                    token_usage.total_tokens
    #                )
    #    return meta
    #
    #def _stage_3(
    #    self, write_dir: str, user: dict, system: dict
    #    ):
    #    """
    #    Stage 3 of IRPD test.
    #    """
    #    # Iterating through cases
    #    meta = {t: 0 for t in system.keys()}
    #    for t in system.keys():
    #        # Checking for requests already made & adjusting user prompt
    #        user_df = user[t]
    #        past_requests = check_completed_requests(
    #            test_dir=write_dir, 
    #            instance_type=t,
    #            stage='3'
    #        )
    #        user_df = user_df[~user_df['window_number'].isin(past_requests)]
    #        if len(user_df) == 0:
    #            break
    #            
    #        # Iterative LLM requests
    #        if self.DIFFICULT_LLM:
    #            system[t] = self._difficult_llms_str(system[t])
    #        system_prompt = system[t]
    #        for i, row in enumerate(user_df.to_dict('records'), start=1):
    #            json_out = False
    #            retries = 0
    #            while not json_out:
    #                # LLM requests
    #                response_out = self._llm_request(
    #                    stage='3',
    #                    user_input=row,
    #                    system_input=system_prompt
    #                )
    #                json_out = self._validate_json_output(
    #                    response=response_out.response,
    #                    stage='3'
    #                )
    #                if not json_out:
    #                    if retries == 3:
    #                        raise TimeoutError("Retries exhuasted")
    #                    print(
    #                        f"Output in stage 3 was not a valid JSON."
    #                        " Retrying..."
    #                    )
    #                    retries += 1
    #            
    #            # Writing request info
    #            write_test(
    #                write_dir=write_dir,
    #                stage='3',
    #                instance_type=t,
    #                system=str(system_prompt),
    #                user=row,
    #                window_number=row['window_number'],
    #                response=response_out.response,
    #                n=i
    #            )
    #            
    #            response_meta = response_out.meta
    #            if i == 1:
    #                meta[t] = response_meta
    #            else:
    #                token_usage = response_meta.usage
    #                meta[t].usage.completion_tokens += (
    #                    token_usage.completion_tokens
    #                )
    #                meta[t].usage.prompt_tokens += (
    #                    token_usage.prompt_tokens
    #                )
    #                meta[t].usage.total_tokens += (
    #                    token_usage.total_tokens
    #                )
    #    return meta
    #
    #def reset_dir_path(self, dir_path: str) -> None:
    #    """Reset the main directory path."""
    #    self.PATH = dir_path
    #
    #def run_test(
    #    self, case: str, ras=None, stages=None, 
    #    treatments=None, test_type: str = 'test', **kwargs
    #    ):
    #    """
    #    Run IRPD test(s).
    #
    #    Args:
    #        case (str): The case type for the summaries used in test.
    #        ras (list[str] | str, optional): The RA or RAs who wrote the 
    #        summaries. Defaults to ['eli', 'thi', 'both'] or ['exp'] if stages 
    #        includes 0. Defaults to None.
    #        stages (list[str] | str, optional): A set or single stage to be 
    #        run. Defaults to ['1', '1r', '1c', '2', '3']. Defaults to None.
    #        treatments (list[str] | str, optional): A set or single treatment 
    #        to be run. Defaults to ['noise', 'no_noise', 'merged']. Defaults 
    #        to None.
    #        test_type (str, optional): The type of test to be run. Defaults 
    #        to 'test'.
    #        **kwargs:
    #            - max_instances (int): Maximum number of instances used in 
    #            Stage 2 and/or 3.
    #            - N (int): Number of times to run test. Can only be greater
    #            than 1 if test_type is replicaiton or cross model_validation.
    #            - llms (list): The LLMs to use for cross-model validation
    #            tests. Must be from valid list of LLMs.
    #            - llm_config (LLMConfig, object): The configuration of the
    #            LLM or LLMs. Must be a LLMConfig object.
    #    """
    #    max_instances = kwargs.get('max_instances', None)
    #    replicaitons = kwargs.get('N', 1)
    #    llm_config = kwargs.get('llm_config', None)
    #    llms = kwargs.get('llms', ['gpt-4o'])
    #    
    #    self._validate_arg(stages, self.VALID_STAGES, "stages")
    #    self._validate_arg(ras, self.VALID_RAS, "ras")
    #    self._validate_arg(llms, self.LLMS, "llms")
    #    self._validate_arg(treatments, self.VALID_TREATMENTS, "treatments")
    #    self._validate_arg([case], self.VALID_CASES, "case")
    #    self._validate_arg([test_type], self._valid_types, "test_type")
    #    
    #    if '0' in stages and ras != ['exp']:
    #        ras = ['exp']
    #        log.warning(f"Note 'ras' defaulted to {ras} since 'stages' include '0'.")
    #    if (test_type == 'subtest' and (len(ras) > 1 or len(treatments) > 1)):
    #        raise ValueError(
    #            "Invalid number of 'ras' or 'treatments' for 'subtest'."
    #            f" Must have length 1. Got: {ras}, {treatments}"
    #        )
    #    if ((len(ras) > 1 or len(treatments) > 1) and not ('0' in stages or '1' in stages)):
    #        raise ValueError(
    #            "If multiple ras or treatments specified, stages must"
    #            f" contain '0' or '1'. Got: {ras}, {treatments}, and {stages}"
    #        )
    #    if (test_type not in {'replication', 'model_validation'} and replicaitons > 1):
    #        raise ValueError(
    #            "The number of replications, 'N', cannot be greater than 1"
    #            " if 'test_type' is not replication or model_validation. Got: "
    #            f"{replicaitons}"
    #        )
    #    
    #    stages = sorted(stages, key=self.VALID_STAGES.index)
    #    ras = sorted(ras, key=self.VALID_RAS.index)
    #    treatments = sorted(treatments, key=self.VALID_TREATMENTS.index)
    #    
    #    # Calculating simple logical variable to determine if replication type
    #    replication_type = False
    #    if test_type in {'replication', 'model_validation'}:
    #        replication_type = True
    #    
    #    # Getting test directories
    #    test_dirs = get_test_directory(
    #        output_dir=self.OUTPATH,
    #        case=case,
    #        test_type=test_type,
    #        stage=stages,
    #        ra_num=len(ras),
    #        treatment_num = len(treatments)
    #    )
    #
    #    # Compressing ras, stages, & treatments
    #    test_zip = list(product(ras, treatments, stages))
    #    
    #    # Initializing test info
    #    for llm in llms:
    #        self.llm = self._initialize_llm(llm, llm_config)
    #        for n in range(1, replicaitons + 1):
    #            for ra, treatment, stage in test_zip:
    #                # Getting the correct directory
    #                test_dir = test_dirs[
    #                    (treatments.index(treatment) * len(ras)) + ras.index(ra)
    #                ]
    #                
    #                if test_type == 'replication':
    #                    write_dir = os.path.join(
    #                        test_dir, f"replication_{n}"
    #                    )
    #                    print(
    #                        f"Running replication {n}/{replicaitons}", end='\r'
    #                    )
    #                elif test_type == 'model_validation':
    #                    write_dir = os.path.join(
    #                        test_dir, llm, f"replication_{n}"
    #                    )
    #                    print(
    #                        f"{llm}: Running replication {n}/{replicaitons}",
    #                        end='\r'
    #                    )
    #                else:
    #                    write_dir = os.path.join(test_dir, "raw")
    #                    print(
    #                        f"[{treatments.index(treatment) * len(ras)
    #                            + ras.index(ra) + 1}/"
    #                        f"{len(treatments) * len(ras)}]: "
    #                        f"Running Stage {stage} "
    #                        f"({stages.index(stage) + 1}/{len(stages)})...."
    #                    )
    #
    #                # Compressed test info
    #                test_info = dict(
    #                    case=case,
    #                    ra=ra,
    #                    treatment=treatment,
    #                    stage=stage
    #                )
    #                
    #                # Gathering prompts
    #                system = get_system_prompt(
    #                    **test_info,
    #                    prompt_dir=self.PROMPTPATH,
    #                    write_dir=write_dir
    #                )
    #                user = get_user_prompt(
    #                    **test_info,
    #                    main_dir=self.PATH,
    #                    write_dir=write_dir,
    #                    max_instances=max_instances,
    #                    replication=replication_type
    #                )
    #                
    #                # Defining 'cases' for uni_siwitch case
    #                cases = get_cases(case=case)
    #                
    #                # Running test
    #                meta = {}
    #                if stage != '1c':
    #                    for i in cases:
    #                        meta[i] = getattr(self,self._test_methods[stage])(
    #                            user=user[i],
    #                            system=system[i],
    #                            write_dir=write_dir
    #                        )
    #                else:
    #                    meta[case] =  getattr(self, self._test_methods[stage])(
    #                        user=user,
    #                        system=system,
    #                        test_info=test_info,
    #                        write_dir=write_dir
    #                    )
    #                
    #                output_dir = test_dir if not replication_type else write_dir
    #                if stage in {'1', '1r', '1c'}:
    #                    json_to_pdf(
    #                        test_dir=output_dir,
    #                        write_dir=write_dir,
    #                        stage=stage,
    #                        case=case
    #                    )
    #                elif stage in {'2', '3'}:
    #                    build_gpt_output(
    #                        **test_info,
    #                        test_dir=output_dir,
    #                        main_dir=self.PATH,
    #                        write_dir=write_dir,
    #                        max_instances=max_instances
    #                    )
    #                write_test_info(
    #                    meta=meta,
    #                    write_dir=write_dir,
    #                    model_info=self.llm.config,
    #                    data_file=test_info,
    #                    stage=stage
    #                )
    #        print()
    #    print(f"Test complete, check {self.OUTPATH} for output.")