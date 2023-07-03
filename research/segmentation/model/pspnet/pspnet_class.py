class PSPNet:
    def __init__(self, args):
        self.args = args
        self.model = self.prepare_model()

