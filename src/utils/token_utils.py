import os 
import functools

def token_loader(filepath: str) -> list[str]: 
    with open(filepath, "r") as r: 
        keys = r.readlines()
    
    keys = [ 
        key.split("\n")[0]
        for key in keys  
    ]    

    return keys


class TokenRotator:
    def __init__(self, token_list: list[str]):
        self.current_token = os.environ["OPENAI_API_KEY"]
        self.token_list = token_list

    def _update_token(self, new_token):
        os.environ["OPENAI_API_KEY"] = new_token
        self.current_token = os.environ["OPENAI_API_KEY"]

    def rotate_token(self):
        new_token = self.token_list.pop(0)
        self.token_list.append(self.current_token)
        self._update_token(new_token)


def token_rotator_decorator(token_rotator):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            token_rotator.rotate_token()  # Rotate the token before calling the function
            return func(*args, **kwargs)
        return wrapper
    return decorator