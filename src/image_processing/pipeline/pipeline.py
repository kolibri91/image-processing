from .block import Block


class Pipeline(Block):
    def __init__(self):
        self.blocks = []

    def add(self, block: Block):
        self.blocks.append(block)

    def __call__(self, input_image):
        result = input_image
        for block in self.blocks:
            result = block(result)
        return result
