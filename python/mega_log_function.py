import logging
from logging.handlers import RotatingFileHandler

def get_logger(self, app_name, log_level):
        log_file = "loader.log"

        # Check the operating system
        current_os = platform.system()

        if current_os == "Windows":
            # Windows log directory
            log_dir = os.path.join(os.getenv("APPDATA"), app_name, "logs")
        elif current_os == "Linux":
            # Linux log directory (user-specific)
            log_dir = os.path.join(os.path.expanduser("~"), ".local", "share", app_name, "logs")
        else:
            # Fallback log directory
            log_dir = os.path.join(os.getcwd(), "logs")

        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, log_file)

        logger = logging.getLogger(app_name)
        # only create the logger if it doesn't already exist
        if not logger.hasHandlers():
            logger.setLevel(log_level)
            # Create handlers
            c_handler = logging.StreamHandler()
            f_handler = RotatingFileHandler(log_path, maxBytes=10**6, backupCount=5)
            c_handler.setLevel(log_level)
            f_handler.setLevel(log_level)

            # Create formatters and add it to handlers
            c_format = logging.Formatter("%(name)s - %(levelname)s - %(message)s")
            f_format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            c_handler.setFormatter(c_format)
            f_handler.setFormatter(f_format)

            # Add handlers to the logger
            logger.addHandler(c_handler)
            logger.addHandler(f_handler)
        return logger
