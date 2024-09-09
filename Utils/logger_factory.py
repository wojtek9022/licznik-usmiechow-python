import logging


class LoggerFactory:
    @staticmethod
    def create_logger(name, log_file_path = 'log.log', level = logging.INFO, log_to_file = True, log_to_console = True):
        logger = logging.getLogger(name)
        logger.setLevel(level)

        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        # Add file handler
        if log_to_file:
            file_handler = logging.FileHandler(log_file_path)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        # Add console handler
        if log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        return logger