from abc import abstractmethod

class kernels():
    def __init__(self):
        pass

    @abstractmethod
    def write_ip(self):
        pass

    @abstractmethod
    def write_create(self):
        pass

    @abstractmethod
    def write_setargs(self):
        pass

    @abstractmethod
    def write_enque(self):
        pass

    @abstractmethod
    def write_release(self):
        pass
