import subprocess
import sys
import logging
from pathlib import Path

from logger import setup_logger
from utils import is_tail_running
from models.irpd_model import IRPDTestClass


log = logging.getLogger("app")



if __name__ == "__main__":
    setup_logger()

    # Optional -- Opens a new terminal/shell w/ live logs
    repo_path = Path(__file__).parents[1]
    if not is_tail_running():
        if sys.platform == "darwin":
            subprocess.Popen([
                "osascript",
                "-e",
                f'tell application "Terminal" to do script "cd {repo_path} && tail -f logs/app.log"'
            ])
        elif sys.platform == "win32":
            subprocess.Popen([
                "start",
                "cmd",
                "/k",
                f'cd /d {repo_path} && tail -f logs/app.log'
            ], shell=True)