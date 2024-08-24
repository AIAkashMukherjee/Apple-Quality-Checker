import os
import logging
from datetime import datetime

# Format datetime for log file name
LOG_FILE = f'{datetime.now().strftime("%m %d %Y %H %M")}.log'

# Define log directory and file paths
log_dir = os.path.join(os.getcwd(), 'logs')
log_file_path = os.path.join(log_dir, LOG_FILE)

# Create log directory if it doesn't exist
os.makedirs(log_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    filename=log_file_path,
    format='[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

logger=logging.getLogger(__name__)

if __name__ == '__main__':
    logging.info('Logging Started')