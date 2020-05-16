from manimlib.imports import *
import math

"""
This fantastic plot is borrowed from:
https://github.com/Elteoremadebeethoven/AnimationsWithManim/blob/master/English/6a_plots_2D/scenes.md#programs

Use the -c flag to change the background color:
manim graphing.py GraphScene Plot -c white

I had to edit the setup_axes function in graph_scene.py to change the color

"""


class Plot(GraphScene):
    CONFIG = {
        "y_max": 4500,
        "y_min": 0,
        "x_max": 40,
        "x_min": 0,
        "x_tick_frequency": 10,
        "y_tick_frequency": 500,
        "axes_color": BLACK,
        "x_axis_label": "Time (s)",
        "y_axis_label": "Power (mW)",
        "x_label_color": BLACK,
        "y_label_color": BLACK,
        "label_nums_color": BLACK
    }

    def construct(self):
        self.setup_axes()
        coords = self.return_coords_from_csv(
            "/home/jonathan/Downloads/kmb_demo-master/csv/KMB_ALL_readings_5")
        path = VMobject()
        set_of_points = self.get_points_from_coords(coords)
        path.set_points_smoothly(set_of_points)
        path.set_color(BLUE)
        self.add(path)
        self.play(
            ShowCreation(path),
            run_time=5
        )
        self.wait()

    def setup_axes(self):
        # Add this line
        GraphScene.setup_axes(self)
        # Parametters of labels
        #   For x
        init_label_x = 0
        end_label_x = 40
        step_x = 5

        #   For y
        init_label_y = 0
        end_label_y = 5000
        step_y = 500
        # Position of labels
        #   For x
        self.x_axis.label_direction = DOWN  # DOWN is default
        #   For y
        self.y_axis.label_direction = LEFT
        # Add labels to graph
        #   For x
        self.x_axis.add_numbers(*range(
            init_label_x,
            end_label_x+step_x,
            step_x
        ))
        #   For y
        self.y_axis.add_numbers(*range(
            init_label_y,
            end_label_y+step_y,
            step_y
        ))
        #   Add Animation
        self.play(
            ShowCreation(self.x_axis),
            ShowCreation(self.y_axis)
        )

    def return_coords_from_csv(self, file_name):
        import csv
        coords = []
        with open(f'{file_name}.csv', 'r') as csvFile:
            reader = csv.reader(csvFile)
            header = next(reader)
            for row in reader:
                x, y = row
                coord = [float(x), float(y)]
                coords.append(coord)
        csvFile.close()
        return coords

    # Covert the data coords to the graph points
    def get_points_from_coords(self, coords):
        return [
            # Convert COORDS -> POINTS
            self.coords_to_point(px, py)
            # See manimlib/scene/graph_scene.py
            for px, py in coords
        ]

    # Return the dots of a set of points
    def get_dots_from_coords(self, coords, radius=0.1):
        points = self.get_points_from_coords(coords)
        dots = VGroup(*[
            Dot(radius=radius).move_to([px, py, pz])
            for px, py, pz in points
        ]
        )
        return dots
