import logging
import sys
from datetime import datetime


class ColoredFormatter(logging.Formatter):
    """Custom formatter with color coding for different log levels"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
    }
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    def format(self, record):
        # Add color to level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{self.BOLD}{levelname}{self.RESET}"
        
        # Format the message
        formatted = super().format(record)
        
        return formatted


def setup_logger(name: str = __name__, level: int = logging.DEBUG) -> logging.Logger:
    """
    Set up a logger with colored console output and detailed formatting.
    
    Args:
        name: Logger name (typically __name__ of the calling module)
        level: Logging level (default: DEBUG)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Console handler with color formatting
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Format: [DATE TIME] [LEVEL] [file:function:line] - message
    format_string = '[%(asctime)s] [%(levelname)s] [%(filename)s:%(funcName)s:%(lineno)d] - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    colored_formatter = ColoredFormatter(format_string, datefmt=date_format)
    console_handler.setFormatter(colored_formatter)
    
    logger.addHandler(console_handler)
    
    return logger
