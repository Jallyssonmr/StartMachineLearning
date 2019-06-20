from sklearn.datasets import load_iris

class DatasetIris(object):
    def __init__(self):
        pass
    
    def get_target(self):
        return load_iris().target
    
    def get_target_names(self):
        return load_iris().target_names 