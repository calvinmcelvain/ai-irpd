import logging
from logger import setup_logger
from testing import *
from get import Get

log = logging.getLogger("app")

TEST_TYPES = [
    "test", "subtest", "replication", "cross_model_validation", "sample_splitting"
]


if __name__ == "__main__":
    setup_logger()
    
