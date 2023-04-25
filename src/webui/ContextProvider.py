from pydantic import BaseModel

# NOTE: un-used right now
class ContextProvider(BaseModel): 
    def __init__(self): 
        self.agent_executor = None

        raise NotImplementedError 



