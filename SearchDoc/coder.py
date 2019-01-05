import varbyte
import simple9


class Coder:
    variant = 'varbyte'

    def __init__(self, variant=None):
        self.variant = variant
        if self.variant is None:
            with open('variant', 'r') as f:
                self.variant = f.read()

    def encode(self, args):
        if self.variant == 'varbyte':
            return varbyte.encode(args)
        else:
            return simple9.encode(args)

    def decode(self, args):
        if self.variant == 'varbyte':
            return varbyte.decode(args)
        else:
            return simple9.decode(args)
