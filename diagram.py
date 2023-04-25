from layers import WrapperLayer
from model import Model

class Diagram:
    def __init__(self, model) -> None:
        self._real_model = model
        self.layers = self._get_layers()
        self.model = Model(self.layers)
    
    def _get_layers(self):
        wrapped_layers = []
        for layer in self._real_model.layers:
            wrapped_layers.append(WrapperLayer(layer))
        return wrapped_layers
    
    def draw(self, ax):
        self.model.draw(ax)
    
    