import logging
import sys
from pathlib import Path


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Настройка логгера с единым форматом для всего проекта
    """
    logger = logging.getLogger(name)
    
    # Если уже настроен, возвращаем существующий
    if logger.hasHandlers():
        return logger
    
    # Настраиваем уровень
    logger.setLevel(level)
    
    # Создаем обработчик для консоли
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Создаем форматтер
    formatter = logging.Formatter(
        "[%(asctime)s] - %(levelname)s [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    
    # Добавляем обработчик
    logger.addHandler(console_handler)
    
    # Отключаем распространение к родительским логгерам
    logger.propagate = False
    
    return logger


def timer(func) -> dict:
    """
    Decorator to measure the execution time of a function.
    param func: The function to be decorated.
    return: dict[str, float] - A dictionary containing the result of the function and its execution time.
    """

    def wrapper(*args, **kwargs):
        import time

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        # Return both the original result and the execution time
        return {"result": result, "execution_time": execution_time}

    return wrapper


