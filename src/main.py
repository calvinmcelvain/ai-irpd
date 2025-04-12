import subprocess
import sys
import logging
from pathlib import Path

from logger import LoggerManager
from helpers.utils import is_tail_running
from core.irpd_model import IRPDTestClass


log = logging.getLogger("app")



if __name__ == "__main__":
    log_manager = LoggerManager()
    log_manager.clear_logs()
    log_manager.setup_logger()

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