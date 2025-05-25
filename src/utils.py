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
        print(f"Execution time: {execution_time:.2f} seconds")
        # Return both the original result and the execution time
        return {"result": result, "execution_time": execution_time}

    return wrapper