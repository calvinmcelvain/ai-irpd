import logging
from logger import setup_logger
from api import app
import uvicorn

log = logging.getLogger(__name__)


if __name__ == "__main__":
    setup_logger()
    uvicorn.run(app, host="0.0.0.0", port=8000)
