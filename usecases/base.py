from abc import ABC, abstractmethod

class UseCase(ABC):
    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def test(self):
        pass