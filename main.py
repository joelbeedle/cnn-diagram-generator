"""
This script is used for visualizing a neural network architecture in 3D using NumPy and Matplotlib. It provides classes and functions to create and manipulate cubes representing network
layers and draw them along with their connections.
"""

import numpy as np
import matplotlib.pyplot as plt
from layers import ConvolutionLayer, BatchNormalization
from model import Model


def main():
    """Main entrypoint for testing"""
    # Example usage
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    layer1 = ConvolutionLayer(3, 3, 64, 64, 1)
    layer2 = BatchNormalization(12, 32, 32)
    layer3 = ConvolutionLayer(12, 12, 32, 32, 1)
    model = Model([layer1, layer2, layer3])
    layer4 = ConvolutionLayer(12, 6, 16, 16, 1)
    model.add(layer4)
    model.draw(ax)

    # Set the aspect ratio of the plot to 'equal'
    limits = np.array([getattr(ax, f"get_{axis}lim")() for axis in "xyz"])
    ax.set_box_aspect(np.ptp(limits, axis=1))

    plt.show()


if __name__ == "__main__":
    main()
