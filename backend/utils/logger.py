import logging
import os
from backend.config import settings

os.makedirs(settings.LOG_DIR, exist_ok=True)
log_file = os.path.join(settings.LOG_DIR, "app.log")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("HybridRAG")