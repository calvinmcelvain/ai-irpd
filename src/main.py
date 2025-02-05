import logging
from logger import setup_logger
from argparse import ArgumentParser
from src.irpd import IRPD
from models.models import LLMModel


log = logging.getLogger(__name__)

if __name__ == "__main__":
    setup_logger()
    
    parser = ArgumentParser(description="Run an IRPD test")
    parser.add_argument(
        "--case",
        type=str,
        choices=IRPD._VALID_CASES,
        required=True,
        help="Case type (e.g., 'uni', 'switch', etc.)."
    )
    parser.add_argument(
        "--ras",
        nargs="+",
        choices=IRPD._VALID_RAS,
        required=True,
        help="List of RAs to use (e.g., 'thi', 'eli', 'both', 'exp')."
    )
    parser.add_argument(
        "--treatments",
        nargs="+",
        choices=IRPD._VALID_TREATMENTS,
        required=True,
        help="Treatment conditions (e.g., 'noise', 'no_noise', 'merged')."
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        choices=IRPD._VALID_STAGES,
        required=True,
        help="Test stages (e.g., '0', '1', '2', etc.)."
    )
    parser.add_argument(
        "--test-type",
        type=str,
        choices=IRPD._VALID_TEST_TYPES,
        default=None,
        help="Type of test to run."
    )
    parser.add_argument(
        "--llm",
        nargs="+",
        required=False,
        choices=LLMModel._member_names_,
        default=None,
        help="The llm used for testing."
    )
    parser.add_argument(
        "--llm-config",
        type=str,
        default=None,
        choices=IRPD._VALID_LLM_CONFIGS,
        help="Configuration for LLM models."
    )
    parser.add_argument(
        "--N",
        type=int,
        default=None,
        help="Number of test repetitions."
    )
    parser.add_argument(
        "--max-instances",
        type=int,
        default=None,
        help="Maximum number of instances to run."
    )
    parser.add_argument(
        "--project-path",
        type=str,
        default=None,
        help="Path to the project directory."
    )
    parser.add_argument(
        "--print-response",
        action="store_true",
        default=None,
        help="Print model responses."
    )
    parser.add_argument(
        "--new-test",
        action="store_true",
        default=None,
        help="Start a new test instead of continuing an existing one."
    )
    args = parser.parse_args()

    # Initialize and run the IRPD test
    test_args = {
        "case": args.case,
        "ras": args.ras,
        "treatments": args.treatments,
        "stages": args.stages,
        "test_type": args.test_type,
        "llms": args.llm,
        "llm_config": args.llm_config,
        "N": args.N,
        "max_instances": args.max_instances,
        "project_path": args.project_path,
        "print_response": args.print_response,
        "new_test": args.new_test
    }
    test_load = {key: value for key, value in test_args.items() if value}
    log.info(f"Arguments: {test_load}")
    irpd_test = IRPD(**test_load)

    irpd_test.run()
