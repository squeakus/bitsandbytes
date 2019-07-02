from manimlib.imports import *
from math import cos, sin, pi
import numpy as np

class MoveTo(Scene):
    def construct(self):
        square = Square() # creating circle mobject
        square.move_to(2*RIGHT+3*UP) # .move_to moves the circle mobject to coordinate: (2,3)
        square.move_to(2*LEFT+3*DOWN) # .move_to moves the circle mobject to coordinate: (2,3)
        self.play(ShowCreation(square)) 
        self.wait(2)

class CircleDraw(Scene):
    def construct(self):
        circle = Circle() # creating circle mobject
        square.move_to(2*RIGHT+3*UP) # .move_to moves the circle mobject to coordinate: (2,3)
        square.move_to(2*LEFT+3*DOWN) # .move_to moves the circle mobject to coordinate: (2,3)
        self.play(ShowCreation(square)) 
        self.wait(2)

class ApplyMethodEx(Scene):
    def construct(self):
        square = Square() # creating circle mobject
        square.move_to(2*RIGHT+3*UP) # .move_to moves the circle mobject to coordinate: (2,3)
        self.play(ShowCreation(square)) # create square mobject on scene
        self.play(ApplyMethod(square.move_to, 4*LEFT+3*DOWN)) # move the mobject
        self.wait(2)

class Sq_to_Cr(Scene):
    def construct(self):
        s1 = Square() # creating square mobject
        c1 = Circle() # creating circle mobject
        self.play(ShowCreation(s1)) 
        self.play(Transform(s1, c1)) # transforming s1 -> c1
        self.wait(2)

class Shapes(Scene):
    def construct(self):
        #######Code#######
        #Making Shapes
        circle = Circle(color=YELLOW)
        square = Square(color=DARK_BLUE)
        square.surround(circle)

        rectangle = Rectangle(height=2, width=3, color=RED)
        ring=Annulus(inner_radius=.2, outer_radius=1, color=BLUE)
        ring2 = Annulus(inner_radius=0.6, outer_radius=1, color=BLUE)
        ring3=Annulus(inner_radius=.2, outer_radius=1, color=BLUE)
        ellipse=Ellipse(width=5, height=3, color=DARK_BLUE)

        pointers = []
        for i in range(8):
            pointers.append(Line(ORIGIN, np.array([cos(pi/180*360/8*i),sin(pi/180*360/8*i), 0]),color=YELLOW))

        #Showing animation
        self.add(circle)
        self.play(FadeIn(square))
        self.play(Transform(square, rectangle))
        self.play(FadeOut(circle), FadeIn(ring))
        self.play(Transform(ring,ring2))
        self.play(Transform(ring2, ring))
        self.play(FadeOut(square), GrowFromCenter(ellipse), Transform(ring2, ring3))
        self.add(*pointers)
        self.wait(2)

class TextCreation(Scene):
  def construct(self):
    display_text = TextMobject("Getting Started with Manim")
    self.play(ShowCreation(display_text))
    self.wait(2)

class TextRotation(Scene):
  def construct(self):
    display_text = TextMobject("Getting Started with Manim")
    self.play(ShowCreation(display_text)) # display text on scene
    
    x = ApplyMethod(display_text.rotate, np.pi/2) # rotate text on screen
    print(x)
    self.play(x)
    self.wait(2)

class MoveText(Scene):
  def construct(self):
    display_text1 = TextMobject("Getting Started with Manim")
    display_text2 = TextMobject(r"TextMobjects with \LaTeX")
    display_text1.move_to(3*UP) # move display_text1 upwards before showing it on scene
    self.add(display_text1) # show display_text1 on the updated location
    self.play(ShowCreation(display_text2)) # show creation of display_text2 at the center
    self.play(ApplyMethod(display_text2.move_to, 2*DOWN+1*LEFT)) # move display_text2 
    self.wait(2)

class RotateAndHighlight(Scene):
    #Rotation of text and highlighting with surrounding geometries
    def construct(self):
        square=Square(side_length=5,fill_color=YELLOW, fill_opacity=1)
        label=TextMobject("Text at an angle")
        label.bg=BackgroundRectangle(label,fill_opacity=1)
        label_group=VGroup(label.bg,label)  #Order matters
        label_group.rotate(TAU/8)
        label2=TextMobject("Boxed text",color=BLACK)
        label2.bg=SurroundingRectangle(label2,color=BLUE,fill_color=RED, fill_opacity=.5)
        label2_group=VGroup(label2,label2.bg)
        label2_group.next_to(label_group,DOWN)
        label3=TextMobject("Rainbow")
        label3.scale(2)
        label3.set_color_by_gradient(RED, ORANGE, YELLOW, GREEN, BLUE, PURPLE)
        label3.to_edge(DOWN)

        self.add(square)
        self.play(FadeIn(label_group))
        self.play(FadeIn(label2_group))
        self.play(FadeIn(label3))

class AddingMoreText(Scene):
    #Playing around with text properties
    def construct(self):
        quote = TextMobject("Imagination is more important than knowledge")
        quote.set_color(RED)
        quote.to_edge(UP)
        quote2 = TextMobject("A person who never made a mistake never tried anything new")
        quote2.set_color(YELLOW)
        author=TextMobject("-Albert Einstein")
        author.scale(0.75)
        author.next_to(quote.get_corner(DOWN+RIGHT),DOWN)

        self.add(quote)
        self.add(author)
        self.wait(2)
        self.play(Transform(quote,quote2),
          ApplyMethod(author.move_to,quote2.get_corner(DOWN+RIGHT)+DOWN+2*LEFT))
        
        self.play(ApplyMethod(author.scale,1.5))
        author.match_color(quote2)
        self.play(FadeOut(quote))

class TextTransformation(Scene):
  def construct(self):
     display_text_1 = TextMobject(r"Getting Started with Manim")
     display_text_2 = TextMobject(r"TextMobject with \LaTeX support | $e^{\pi i} = -1$")
     # transform display_text_1 to display_text_2 on the scene
     self.play(Transform(display_text_1, display_text_2)) 
     self.wait(2)

class ColoringEquations(Scene):
    #Grouping and coloring parts of equations
    def construct(self):
        line1=TexMobject(r"\text{The vector } \vec{F}_{net} \text{ is the net }",r"\text{force }",r"\text{on object of mass }")
        line1.set_color_by_tex("force", BLUE)
        line2=TexMobject("m", "\\text{ and acceleration }", "\\vec{a}", ".  ")
        line2.set_color_by_tex_to_color_map({
            "m": YELLOW,
            "{a}": RED
        })
        sentence=VGroup(line1,line2)
        sentence.arrange_submobjects(DOWN, buff=MED_LARGE_BUFF)
        self.play(Write(sentence))

class PlotFunctions(GraphScene):
    CONFIG = {
            "graph_origin" : ORIGIN ,
            "function_color" : RED ,
            "axes_color" : GREEN,
            "x_labeled_nums" :range(-10,12,2),
            "x_min": -1,
            "x_max": 10,
            "x_axis_width": 9,
            "x_tick_frequency": 1,
            "x_leftmost_tick": None, # Change if different from x_min
            "x_labeled_nums": None,
            "x_axis_label": "$x$",
            "y_min": -1,
            "y_max": 10,
            "y_axis_height": 6,
            "y_tick_frequency": 1,
            "y_bottom_tick": None, # Change if different from y_min
            "y_labeled_nums": None,
            "y_axis_label": "$y$",
            "axes_color": GREY,
            "graph_origin": 2.5 * DOWN + 4 * LEFT,
            "exclude_zero_label": True,
            "num_graph_anchor_points": 25,
            "default_graph_colors": [BLUE, GREEN, YELLOW],
            "default_derivative_color": GREEN,
            "default_input_color": YELLOW,
            "default_riemann_start_color": BLUE,
            "default_riemann_end_color": GREEN,
            "area_opacity": 0.8,
            "num_rects": 50,
    }

    def construct(self):
        self.setup_axes(animate=True)
        func_graph=self.get_graph(self.func_to_graph,self.function_color)
        func_graph2=self.get_graph(self.func_to_graph2)
        vert_line = self.get_vertical_line_to_graph(TAU,func_graph,color=YELLOW)
        graph_lab = self.get_graph_label(func_graph, label = "\\cos(x)")
        graph_lab2=self.get_graph_label(func_graph2,label = "\\sin(x)", x_val=-10, direction=UP/2)
        two_pi = TexMobject("x = 2 \\pi")
        label_coord = self.input_to_graph_point(TAU,func_graph)
        two_pi.next_to(label_coord,RIGHT+UP)

        self.play(ShowCreation(func_graph),ShowCreation(func_graph2))
        self.play(ShowCreation(vert_line), ShowCreation(graph_lab), ShowCreation(graph_lab2),ShowCreation(two_pi))

    def func_to_graph(self,x):
        return np.cos(x)

    def func_to_graph2(self,x):
        return np.sin(x)

class DrawAnAxis(Scene):
    CONFIG = { "plane_kwargs" : { 
        "x_line_frequency" : 2,
        "y_line_frequency" :2
        }
    }

    def construct(self):
        my_plane = NumberPlane(**self.plane_kwargs)
        my_plane.add(my_plane.get_axis_labels())
        self.add(my_plane)
        self.play(ShowCreation(my_plane))

class ExampleApproximation(GraphScene):
    CONFIG = {
        "function" : lambda x : np.cos(x), 
        "function_color" : BLUE,
        "taylor" : [lambda x: 1, lambda x: 1-x**2/2, lambda x: 1-x**2/math.factorial(2)+x**4/math.factorial(4), lambda x: 1-x**2/2+x**4/math.factorial(4)-x**6/math.factorial(6),
        lambda x: 1-x**2/math.factorial(2)+x**4/math.factorial(4)-x**6/math.factorial(6)+x**8/math.factorial(8), lambda x: 1-x**2/math.factorial(2)+x**4/math.factorial(4)-x**6/math.factorial(6)+x**8/math.factorial(8) - x**10/math.factorial(10)],
        "center_point" : 0,
        "approximation_color" : GREEN,
        "x_min" : -10,
        "x_max" : 10,
        "y_min" : -1,
        "y_max" : 1,
        "graph_origin" : ORIGIN ,
        "x_labeled_nums" :range(-10,12,2),

    }
    def construct(self):
        self.setup_axes(animate=True)
        func_graph = self.get_graph(
            self.function,
            self.function_color,
        )
        approx_graphs = [
            self.get_graph(
                f,
                self.approximation_color
            )
            for f in self.taylor
        ]

        term_num = [
            TexMobject("n = " + str(n),aligned_edge=TOP)
            for n in range(0,8)]
        #[t.to_edge(BOTTOM,buff=SMALL_BUFF) for t in term_num]


        #term = TexMobject("")
        #term.to_edge(BOTTOM,buff=SMALL_BUFF)
        term = VectorizedPoint(3*DOWN)

        approx_graph = VectorizedPoint(
            self.input_to_graph_point(self.center_point, func_graph)
        )

        self.play(
            ShowCreation(func_graph),
        )
        for n,graph in enumerate(approx_graphs):
            self.play(
                Transform(approx_graph, graph, run_time = 2),
                Transform(term,term_num[n])
            )
            self.wait()

class FirstScene(Scene): # All classes that animate are a subclass of scene
	def construct(self):
		"""
		All the animation code must lie within construct method.
		This method is invoked by manim.py
		"""
		circle = Circle() # creating circle object
		self.add(circle) # adding that circle object to the scene
		self.wait(2) # waiting for two second ~ time.sleep(2)