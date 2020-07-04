from abc import ABCMeta, abstractmethod


class Block(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, input_image):
        pass
