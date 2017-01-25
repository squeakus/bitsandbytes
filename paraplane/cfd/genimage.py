try: paraview.simple
except: from paraview.simple import *
paraview.simple._DisableFirstRenderCameraReset()


cfd_OpenFOAM = PV3FoamReader( FileName='/home/jonathan/Jonathan/programs/planegen/cfd/cfd.OpenFOAM' )

cfd_OpenFOAM = GetActiveSource()
cfd_OpenFOAM.VolumeFields = ['p', 'U']
cfd_OpenFOAM.MeshParts = ['planeGroup - group']

RenderView1 = GetRenderView()
DataRepresentation1 = Show()
DataRepresentation1.EdgeColor = [0.0, 0.0, 0.5000076295109483]


a1_p_PVLookupTable = GetLookupTableForArray( "p", 1, NanColor=[0.498039, 0.498039, 0.498039], RGBPoints=[-163.65480041503906, 0.0, 0.0, 1.0, 168.36163330078125, 1.0, 0.0, 0.0], VectorMode='Magnitude', ColorSpace='HSV', ScalarRangeInitialized=1.0 )

a1_p_PiecewiseFunction = CreatePiecewiseFunction()

RenderView1.CameraPosition = [2.140625, 0.0, 13.360863132468259]
RenderView1.CameraFocalPoint = [2.140625, 0.0, 3.0]
RenderView1.CameraClippingRange = [9.573192001143576, 11.377369829455283]
RenderView1.CenterOfRotation = [2.140625, 0.0, 3.0]
RenderView1.CameraParallelScale = 2.6815887023833467

DataRepresentation1.ColorArrayName = 'p'
DataRepresentation1.LookupTable = a1_p_PVLookupTable

cfd_OpenFOAM.VolumeFields = ['p', 'U']
cfd_OpenFOAM.PointFields = []
cfd_OpenFOAM.LagrangianFields = []
cfd_OpenFOAM.MeshParts = ['internalMesh']

cfd_OpenFOAM.UiRefresh = 0

cfd_OpenFOAM.VolumeFields = ['p', 'U']
cfd_OpenFOAM.MeshParts = ['internalMesh']

SetActiveSource(cfd_OpenFOAM)
Slice1 = Slice( SliceType="Plane" )

SetActiveSource(cfd_OpenFOAM)
DataRepresentation2 = Show()
DataRepresentation2.ScalarOpacityUnitDistance = 0.25906362137240246
DataRepresentation2.Representation = 'Outline'
DataRepresentation2.EdgeColor = [0.0, 0.0, 0.5000076295109483]

RenderView1.CameraClippingRange = [1.3072545011435759, 19.521276079455284]

Slice1.SliceOffsetValues = [0.0]
Slice1.SliceType.Origin = [5.0, 0.0, 4.0]
Slice1.SliceType = "Plane"

RenderView1.CameraClippingRange = [1.3072545011435759, 19.521276079455284]

cfd_OpenFOAM.VolumeFields = ['nut']
cfd_OpenFOAM.MeshParts = ['internalMesh']

DataRepresentation2.EdgeColor = [0.0, 0.0, 0.5019607843137255]

Slice1.SliceType.Normal = [0.0, 1.0, 0.0]

SetActiveSource(Slice1)
DataRepresentation3 = Show()
DataRepresentation3.EdgeColor = [0.0, 0.0, 0.5019607843137255]

a1_nut_PVLookupTable = GetLookupTableForArray( "nut", 1, NanColor=[0.498039, 0.498039, 0.498039], RGBPoints=[9.656450856709853e-05, 0.0, 0.0, 1.0, 0.1348309963941574, 1.0, 0.0, 0.0], VectorMode='Magnitude', ColorSpace='HSV', ScalarRangeInitialized=1.0 )

a1_nut_PiecewiseFunction = CreatePiecewiseFunction()

RenderView1.CameraViewUp = [0.0044801117140750866, 0.43072324745370627, 0.9024729429195996]
RenderView1.CameraPosition = [2.1970476358107205, -9.350462708196229, 7.462415094827063]
RenderView1.CameraClippingRange = [0.03423088664834008, 34.230886648340075]

DataRepresentation3.ColorArrayName = 'nut'
DataRepresentation3.LookupTable = a1_nut_PVLookupTable

WriteImage('/home/jonathan/Jonathan/programs/planegen/cfd/moo.png')


Render()
exit()
