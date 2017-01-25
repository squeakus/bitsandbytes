import sys, random
sys.path.append('/usr/lib/freecad/lib')
import FreeCAD
import Part
from FreeCAD import Base
import Sketcher
import Mesh

def main():    
    slices = parse_car('car_slices.csv')
    generate_car('gencar', slices)

def parse_car(filename):
    infile = open(filename,'r')
    sliceindexes = []
    slices = []
    tmparray = []

    for line in infile:
        if not line.startswith(','):
            line = line.rstrip('\r\n')
            line = line.split(',')
            x,y,z = float(line[0]), float(line[1]), float(line[2])

            if not x in sliceindexes:
                sliceindexes.append(x)
                slices.append(tmparray)
                tmparray = []

            tmparray.append([x,y,z])
    #this little catch is required to append the last set
    slices.append(tmparray)
    print "number of slices", len(slices)
    return slices[1:]

def generate_car(carname, slices):
    docname = carname
    sketchnames = []
    App.newDocument(docname)
    App.setActiveDocument(docname)
    App.ActiveDocument=App.getDocument(docname)

    for i,s in enumerate(slices):
        offset = s[0][0]
        sname = "sketch"+str(i)
        print sname, "offset", offset

        sketchnames.append(sname)
        App.activeDocument().addObject('Sketcher::SketchObject',sname)
        App.activeDocument().getObject(sname).Placement = App.Placement(App.Vector(offset,0.000000,0.000000),
                                                                        App.Rotation(0.500000,0.500000,0.500000,0.500000))

        for j in range(len(s)-1):
            x1,y1,z1 = s[j][0], s[j][1], s[j][2]
            x2,y2,z2 = s[j+1][0], s[j+1][1], s[j+1][2]
            App.ActiveDocument.getObject(sname).addGeometry(Part.Line(App.Vector(y1,z1,0),App.Vector(y2,z2,0)))
        App.ActiveDocument.recompute()



    #loft them together
    #App.getDocument(carname).addObject('Part::Loft','Loft')
    #App.getDocument(carname).getObject('Loft').Sections=[App.getDocument(carname).sketch54,App.getDocument(carname).sketch53,App.getDocument(carname).sketch52,App.getDocument(carname).sketch51,App.getDocument(carname).sketch50,App.getDocument(carname).sketch49,App.getDocument(carname).sketch48,App.getDocument(carname).sketch47,App.getDocument(carname).sketch46,App.getDocument(carname).sketch45,App.getDocument(carname).sketch44,App.getDocument(carname).sketch43,App.getDocument(carname).sketch42,App.getDocument(carname).sketch41,App.getDocument(carname).sketch40,App.getDocument(carname).sketch39,App.getDocument(carname).sketch38,App.getDocument(carname).sketch37,App.getDocument(carname).sketch36,App.getDocument(carname).sketch35,App.getDocument(carname).sketch34,App.getDocument(carname).sketch33,App.getDocument(carname).sketch32,App.getDocument(carname).sketch31,App.getDocument(carname).sketch30,App.getDocument(carname).sketch29,App.getDocument(carname).sketch28,App.getDocument(carname).sketch27,App.getDocument(carname).sketch26,App.getDocument(carname).sketch25,App.getDocument(carname).sketch24,App.getDocument(carname).sketch23,App.getDocument(carname).sketch22,App.getDocument(carname).sketch21,App.getDocument(carname).sketch20,App.getDocument(carname).sketch19,App.getDocument(carname).sketch18,App.getDocument(carname).sketch17,App.getDocument(carname).sketch16,App.getDocument(carname).sketch15,App.getDocument(carname).sketch15,App.getDocument(carname).sketch14,App.getDocument(carname).sketch13,App.getDocument(carname).sketch12,App.getDocument(carname).sketch11,App.getDocument(carname).sketch10,App.getDocument(carname).sketch9, App.getDocument(carname).sketch8, App.getDocument(carname).sketch7, App.getDocument(carname).sketch6, App.getDocument(carname).sketch5, App.getDocument(carname).sketch4, App.getDocument(carname).sketch3, App.getDocument(carname).sketch2, App.getDocument(carname).sketch1,]

    #App.getDocument(carname).getObject('Loft').Solid=True
    #App.getDocument(carname).getObject('Loft').Ruled=True
    #App.getDocument(carname).getObject('Loft').Ruled=False
    #App.ActiveDocument.recompute()

    #__doc__=FreeCAD.getDocument(docname)
    #__doc__.addObject("Part::Mirroring")
    #__doc__.ActiveObject.Source=__doc__.getObject("Loft")
    #__doc__.ActiveObject.Label="Loft (Mirror #1)"
    #__doc__.ActiveObject.Normal=(1,0,0)
    #__doc__.ActiveObject.Base=(0,0,0)
    #del __doc__
    #App.ActiveDocument.recompute()

    App.getDocument(docname).saveAs(carname+'.fcstd')

    #__objs__=[]
    #__objs__.append(FreeCAD.getDocument(carname).getObject("Loft"))
    #__objs__.append(FreeCAD.getDocument(carname).getObject("Part__Mirroring"))
    # Mesh.export(__objs__,'./'+carname+".stl")
    del __objs__
        
            
def old_generate_car(carname, slices):
    docname = carname
    sketchnames = []
    App.newDocument(docname)
    App.setActiveDocument(docname)
    App.ActiveDocument=App.getDocument(docname)
            
    for i in range(10):
        offset = i * 5
        sname = "sketch"+str(i)
        sketchnames.append(sname)
        App.activeDocument().addObject('Sketcher::SketchObject',sname)
        App.activeDocument().getObject(sname).Placement = App.Placement(App.Vector(offset,0.000000,0.000000),App.Rotation(0.500000,0.500000,0.500000,0.500000))
        App.ActiveDocument.getObject(sname).addGeometry(Part.Line(App.Vector(0.000000,0.000000,0),App.Vector(30.000000,15.000000,0)))
        #App.ActiveDocument.getObject(sname).addConstraint(Sketcher.Constraint('Coincident',-1,1,0,1)) 
        App.ActiveDocument.getObject(sname).addGeometry(Part.Line(App.Vector(30.000000,15.000000,0),App.Vector(50.000000,15.000000,0)))
        #App.ActiveDocument.getObject(sname).addConstraint(Sketcher.Constraint('Coincident',0,2,1,1)) 
        #App.ActiveDocument.getObject(sname).addConstraint(Sketcher.Constraint('Horizontal',1)) 
        App.ActiveDocument.recompute()
        #App.ActiveDocument.getObject(sname).addGeometry(Part.Line(App.Vector(50.000000,15.000000,0),App.Vector(71.760025,22.753183,0)))
        y = (random.random()*10) + 12
        App.ActiveDocument.getObject(sname).addGeometry(Part.Line(App.Vector(50.000000,15.000000,0),App.Vector(71.760025,y,0)))

        #App.ActiveDocument.getObject(sname).addConstraint(Sketcher.Constraint('Coincident',1,2,2,1)) 
        App.ActiveDocument.getObject(sname).addGeometry(Part.Line(App.Vector(71.760025,y,0),App.Vector(93.929802,26.837082,0)))
        #App.ActiveDocument.getObject(sname).addConstraint(Sketcher.Constraint('Coincident',2,2,3,1)) 
        App.ActiveDocument.getObject(sname).addGeometry(Part.Line(App.Vector(93.929802,26.837082,0),App.Vector(113.765884,28.879032,0)))
        #App.ActiveDocument.getObject(sname).addConstraint(Sketcher.Constraint('Coincident',3,2,4,1)) 
        App.ActiveDocument.getObject(sname).addGeometry(Part.Line(App.Vector(113.765884,28.879032,0),App.Vector(113.474182,0.000000,0)))

        #App.ActiveDocument.getObject(sname).addConstraint(Sketcher.Constraint('Coincident',4,2,5,1)) 
        App.ActiveDocument.getObject(sname).addConstraint(Sketcher.Constraint('Vertical',5)) 
        App.ActiveDocument.getObject(sname).addGeometry(Part.Line(App.Vector(113.765884,0.000000,0),App.Vector(0.000000,-0.000000,0)))
        App.ActiveDocument.getObject(sname).addConstraint(Sketcher.Constraint('Coincident',5,2,6,1)) 
        App.ActiveDocument.getObject(sname).addConstraint(Sketcher.Constraint('Coincident',6,2,0,1)) 
        App.ActiveDocument.getObject(sname).addConstraint(Sketcher.Constraint('Horizontal',6))
        App.ActiveDocument.recompute()

    #loft them together
    App.getDocument(carname).addObject('Part::Loft','Loft')
    App.getDocument(carname).getObject('Loft').Sections=[App.getDocument(carname).sketch9, App.getDocument(carname).sketch8, App.getDocument(carname).sketch7, App.getDocument(carname).sketch6, App.getDocument(carname).sketch5, App.getDocument(carname).sketch4, App.getDocument(carname).sketch3, App.getDocument(carname).sketch2, App.getDocument(carname).sketch1, App.getDocument(carname).sketch0, ]

    #sections = [App.getDocument(carname).getProperty(sname) for sname in sketchnames]
    App.getDocument(carname).getObject('Loft').Sections = sections
    App.getDocument(carname).getObject('Loft').Solid=True
    #App.getDocument(carname).getObject('Loft').Ruled=True
    App.getDocument(carname).getObject('Loft').Ruled=False
    App.ActiveDocument.recompute()

    __doc__=FreeCAD.getDocument(docname)
    __doc__.addObject("Part::Mirroring")
    __doc__.ActiveObject.Source=__doc__.getObject("Loft")
    __doc__.ActiveObject.Label="Loft (Mirror #1)"
    __doc__.ActiveObject.Normal=(1,0,0)
    __doc__.ActiveObject.Base=(0,0,0)
    del __doc__
    App.ActiveDocument.recompute()

    App.getDocument(docname).saveAs(carname+'.fcstd')

    __objs__=[]
    __objs__.append(FreeCAD.getDocument(carname).getObject("Loft"))
    __objs__.append(FreeCAD.getDocument(carname).getObject("Part__Mirroring"))
    Mesh.export(__objs__,'./'+carname+".stl")
    #del __objs__

    
if __name__=="__main__":
    main()
