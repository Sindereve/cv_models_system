REGISTRY = {}

def register(name):
    def wrapper(cls):
        REGISTRY[name] = cls
        return cls
    return wrapper