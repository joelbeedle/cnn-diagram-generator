from layers import WrapperLayer
from model import Model


class Diagram:
    def __init__(self, model) -> None:
        self._real_model = model
        self.layers = self._get_layers()
        self.model = Model(self.layers)

    def _get_layers(self):
        wrapped_layers = []
        idx = 0
        for layer in self._real_model.layers:
            if layer == self._real_model.layers[0]:
                wrapped_layers.append(WrapperLayer(None, layer))
            else:
                wrapped_layers.append(WrapperLayer(wrapped_layers[idx - 1], layer))
            print(wrapped_layers[idx].name)
            idx += 1
        return wrapped_layers

    def draw(self, ax):
        self.model.draw(ax)
