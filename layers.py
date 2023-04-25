class Layer:
    """
    Represents a layer in a neural network model.

    Attributes:
        name (str): The name of the layer.
        input_channels (int): The number of input channels.
        output_channels (int): The number of output channels.
        depth (float): The depth of the layer cube.
        color (str): The color of the layer cube.
        alpha (float): The transparency of the layer cube.
    """

    def __init__(self, name, input_channels, output_channels, depth, color,
                 alpha) -> None:
        self.name = name
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.depth = depth
        self.color = color
        self.alpha = alpha

