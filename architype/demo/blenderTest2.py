from geometry import * 
import graph
shape = graph.graph()
map_mapply([lambda x : [shape.connect((8, 4, 1), x)],lambda x : [dropPerpendicular(x, 2)],lambda x : connect3(x, 2),lambda x : [shape.connect((1, 5, 6), x)],lambda x : [shape.connect((13, 6, 8), x)]], map(lambda t : bezier_form(t, ((4, 1, 0), (2, 14, 5), (12, 10, 7), (10, 10, 15))), make_scalar_list(10)))
shape.create_mesh('test.mesh')
return shape

