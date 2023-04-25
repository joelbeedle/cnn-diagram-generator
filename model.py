from layers import Layer
from cube import Cube

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
                self._draw_connections(cube, self._cubes[index + 1], axis)

    def _draw_connections(self, outbound: Cube, inbound: Cube, axis):
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
