"""
This script is used for visualizing a neural network architecture in 3D using NumPy and Matplotlib. It provides classes 
and functions to create and manipulate cubes representing network layers and draw them along with their connections.
"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def standard_vertices_translations(x, y, z):
    """
    Translate the standard unit cube vertices by the given x, y, and z values.

    Args:
        x (float): The x-coordinate translation.
        y (float): The y-coordinate translation.
        z (float): The z-coordinate translation.

    Returns:
        np.array: A NumPy array containing the translated vertices.
    """
    standard_vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                                  [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
                                 dtype=np.float32)
    new_vertices = []
    for (x1, y1, z1) in standard_vertices:
        new_vertices.append([x1 + x, y1 + y, z1 + z])

    return np.array(new_vertices, dtype=np.float32)


def translate_vertices(old_vertices, new_coords):
    """
    Translate a set of vertices by the given new coordinates.

    Args:
        old_vertices (np.array): A NumPy array containing the original vertices.
        new_coords (tuple): A tuple containing the x, y, and z translation values.

    Returns:
        np.array: A NumPy array containing the translated vertices.
    """
    return old_vertices + np.array(new_coords, dtype=np.float32)


class CubeCollection:
    """
    A collection of Cube objects, used to manage and draw multiple cubes together.

    Attributes:
        cubes (list): A list of Cube objects.
    """

    def __init__(self, cubes=[]):
        self.cubes = cubes

    def add(self, cube: 'Cube'):
        """
        Add a Cube object to the CubeCollection.

        Args:
            cube (Cube): The Cube object to be added to the collection.
        """
        self.cubes.append(cube)

    def draw_cubes(self, ax):
        """
        Draw all the cubes in the collection on the provided 3D axis.

        Args:
            ax (Axes3D): The 3D axis on which to draw the cubes.
        """ 
        for cube in self.cubes:
            cube.draw(ax)


cube_collection = CubeCollection()


class Cube:
    """
    Represents a 3D cube, used to visualize network layers.

    Attributes:
        color (str): The color of the cube.
        alpha (float): The transparency of the cube.
        width (float): The width of the cube.
        height (float): The height of the cube.
        depth (float): The depth of the cube.
        x (float): The x-coordinate of the bottom left vertex of the cube.
        y (float): The y-coordinate of the bottom left vertex of the cube.
        z (float): The z-coordinate of the bottom left vertex of the cube.
        vertices (np.array): The vertices of the cube after scaling and translation.
    """

    def __init__(self, width, height, depth, x, y, z, colour, alpha) -> None:
        self.color = colour
        self.alpha = alpha
        self.__width = width
        self.__height = height
        self.__depth = depth
        self.x = x
        self.y = y
        self.z = z
        cube_collection.add(self)

    def draw(self, ax):
        """
        Draw the cube on the provided 3D axis.

        Args:
                ax (Axes3D): The 3D axis on which to draw the cube.
        """
        standard_vertices = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
                                      [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
                                     dtype=np.float32)

        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4),
                 (0, 4), (1, 5), (2, 6), (3, 7)]

        faces = [(0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (1, 2, 6, 5), (2, 3, 7, 6),
                 (3, 0, 4, 7)]
        temp_vertices = []
        for (x1, y1, z1) in standard_vertices:
            scaled = [x1 * self.width, y1 * self.depth, z1 * self.height]
            temp_vertices.append(scaled)
        pre_translated = np.array(temp_vertices, dtype=np.float32)
        translated = translate_vertices(pre_translated,
                                        (self.x, self.y, self.z))

        self.vertices = np.array(translated, dtype=np.float32)
        # Plot the edges of the cube
        for edge in edges:
            ax.plot(self.vertices[edge, 0], self.vertices[edge, 1],
                    self.vertices[edge, 2], 'black')

        # Create the polygons for each face
        polygons = [
            Poly3DCollection([self.vertices[list(face)]],
                             alpha=self.alpha,
                             facecolor=self.color) for face in faces
        ]

        # Add the polygons to the plot
        for poly in polygons:
            ax.add_collection3d(poly)

    @property
    def width(self):
        return self.__width

    @property
    def height(self):
        return self.__height

    @property
    def depth(self):
        return self.__depth

def draw_connections(outbound: Cube, inbound: Cube, axis):
    """Draws connections between two adjacent cubes, with each four
    corners of the faces of the cubes that face each other connected.

    Args:
        outbound (Cube): The cube from which the connection is outbound.
        inbound (Cube): The cube to which the connection is inbound.
        axis: The matplotlib 3D subplot 
    """
    def get_vertices(from_vertices, to_vertices):
        """Extracts vertices from `from_vertices` and `to_vertices`.

        Given two sets of vertices `from_vertices` and `to_vertices`,
        returns two subsets of vertices consisting of the four vertices of
        `from_vertices` that are on the face of the cube facing `to_vertices`,
        and the four vertices of `to_vertices` that are on the face of the cube
        facing `from_vertices`.

        Args:
            from_vertices (np.ndarray): The vertices of the first (outbound) cube 
            to_vertices (np.ndarray): The vertices of the second (inbound) cube 

        Returns:
            tuple: A tuple of two lists of vertices. The first list contains
                the four vertices of `from_vertices` that are on the face of
                the cube facing the cube of `to_vertices`. The second list
                contains the four vertices of `to_vertices` that are on the
                face of the cube facing `from_vertices`.
        """
        o = [2, 3, 6, 7]
        i = [0, 1, 4, 5]
        from_vertices_list = from_vertices.tolist()
        to_vertices_list = to_vertices.tolist()
        correct_from_vertices = [from_vertices_list[n] for n in o]
        correct_to_vertices = [to_vertices_list[n] for n in i]
        return correct_from_vertices, correct_to_vertices

    out_vertices, in_vertices = get_vertices(outbound.vertices,
                                             inbound.vertices)
    connecting_edges = [(0, 1), (1, 0), (2, 3), (3, 2)]

    for edge in connecting_edges:
        x_out, x_in = out_vertices[edge[0]][0], in_vertices[edge[1]][0]
        y_out, y_in = out_vertices[edge[0]][1], in_vertices[edge[1]][1]
        z_out, z_in = out_vertices[edge[0]][2], in_vertices[edge[1]][2]
        axis.plot([x_out, x_in], [y_out, y_in], [z_out, z_in], 'black')




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


class Model:
    """
    Represents a neural network model, with a list of layers.

    Attributes:
        layers (list): A list of Layer objects.
        _cubes (list): A list of Cube objects representing the layers in the model.
    """

    def __init__(self, layers: list[Layer]):

        def get_color(name):
            name_map = {'Convolution': 'r', 'BatchNormalization': 'b'}
            return name_map[name]

        self.layers = layers
        self._cubes: list[Cube] = []
        for layer in layers:
            self._cubes.append(
                Cube(layer.input_channels, layer.input_channels, layer.depth,
                     0, 0, 0, get_color(layer.name), layer.alpha))

        # We can leave the largest cube where it is, as it is the largest.
        # We need to move others up in the graph so that they are central
        # Find the largest cube
        largest_cube = max(self._cubes,
                           key=lambda cube: cube.width * cube.height)
        prev_cube = None
        # Translate the other cubes to align their center with the largest cube's center
        for i, cube in enumerate(self._cubes):
            if cube != largest_cube:
                # Calculate the offset to translate the cube
                x_offset = (largest_cube.width - cube.width) / 2
                z_offset = (largest_cube.height - cube.height) / 2
                y_offset = 0  # Assuming all cubes are on the same z-plane

                if prev_cube != None:
                    y_offset = prev_cube.depth + prev_cube.y + 10  # Placeholder figure for distance apart
                # Translate the cube
                cube.x += x_offset
                cube.y += y_offset
                cube.z += z_offset
            prev_cube = cube

    def draw(self, axis):
        """
        Draw the model's layers (cubes) and their connections on the provided 3D axis.

        Args:
                axis (Axes3D): The 3D axis on which to draw the layers and connections.
    """
        for cube in self._cubes:
            cube.draw(axis)
        for index, cube in enumerate(self._cubes):
            # Only draw connection if there is a layer next in the model
            if index != len(self._cubes) - 1:
                draw_connections(cube, self._cubes[index + 1], axis)

def main():
    # Example usage
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    #cube1 = Cube(5,5,1, 0, 0, 0, 'r', 0.5)
    #cube2 = Cube(3,3,4, 1, 3, 1, 'b', 0.5)
    layer1 = Layer('Convolution', 64, 64, 5, 'r', 0.5)
    layer2 = Layer('BatchNormalization', 64, 64, 32, 'b', 0.5)
    layer3 = Layer('Convolution', 32, 32, 5, 'b', 0.5)
    model = Model([layer1, layer2, layer3])
    model.draw(ax)

    #cube_collection.draw_cubes(ax)
    #draw_connections(cube1, cube2, ax)

    # Set the aspect ratio of the plot to 'equal'
    limits = np.array([getattr(ax, f'get_{axis}lim')() for axis in 'xyz'])
    ax.set_box_aspect(np.ptp(limits, axis=1))

    plt.show()

if __name__=="__main__":
   main()
