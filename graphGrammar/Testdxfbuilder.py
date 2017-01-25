# WriteDXFPolygon creates a minimal DXF file that only contains
# the ENTITIES section. This subroutine requires five parameters,
# the DXF file name, the number of sides for the polygon, the X
# and Y coordinates for the bottom end of the right-most side
# (it starts in a vertical direction), and the length for each
# side. Note that because this only requests 2D points, it does
# not include the Z coordinates (codes 30 and 31). The lines are
# placed on the layer "Polygon."
filename = 'WriteDXFPolygon.dxf'

DXF = file(filename,'w')
#dxfFile As String, iSides As Integer, _
#dblX As Double, dblY As Double, dblLen As Double)
#Dim i As Integer
#Dim dblA1, dblA, dblPI, dblNX, dblNY As Double
#Open dxfFile For Output As #1
DXF.write('1, 0')
DXF.write('1, "SECTION"')
DXF.write('1, 2')
DXF.write('1, "ENTITIES"')
dblPI = Atn(1) * 4
dblA1 = (2 * dblPI) / iSides
dblA = dblPI / 2
for i = 1 To iSides
    DXF.write('1, 0')
    DXF.write('1, "LINE"')
    DXF.write('1, 8')
    DXF.write('1, "Polygon"')
    DXF.write('1, 10')
    DXF.write('1, dblX')
    DXF.write('1, 20')
    DXF.write('1, dblY')
    dblNX = dblLen * Cos(dblA) + dblX
    dblNY = dblLen * Sin(dblA) + dblY
    DXF.write('1, 11')
    DXF.write('1, dblNX')
    DXF.write('1, 21')
    DXF.write('1, dblNY')
    dblX = dblNX
    dblY = dblNY
    dblA = dblA + dblA1
#Next i
DXF.write('1, 0')
DXF.write('1, "ENDSEC"')
#Writing a DXF Interface Program | 233
DXF.write('1, 0')
DXF.write('1, "EOF"')
DXF.write('1')
DXF.close
