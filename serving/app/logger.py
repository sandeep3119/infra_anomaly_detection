import logging
import json
from .config import settings


STANDARD_LOG_KEYS = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "taskName"
}
class JsonFormatter(logging.Formatter):
    
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "logger": record.name,
            "filename": record.filename,
            "line": record.lineno,
        }
        # extra fields land as top-level attributes on record
        for key, value in record.__dict__.items():
            if key not in logging.LogRecord.__dict__ and key not in STANDARD_LOG_KEYS:
                log_record[key] = value
        return json.dumps(log_record)
    

def setup_logger(name, level=settings.log_level):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.propagate = False

    return logger