import itk
import vtk
import os

dicoms_dir = 'data/dcms'
image_3d_path = 'data/3d_image.mha'
image_3d_seg_path = 'data/3d_image_seg.mha'


def DICOMs_to_3Dimage(data_dir, out_file):
    namesGenerator = itk.GDCMSeriesFileNames.New()
    namesGenerator.SetUseSeriesDetails(True)
    namesGenerator.SetGlobalWarningDisplay(False)
    namesGenerator.SetDirectory(data_dir)

    seriesUID = namesGenerator.GetSeriesUIDs()

    assert len(seriesUID), "No DICOMs in: " + data_dir

    seriesIdentifier = seriesUID[0]
    fileNames = namesGenerator.GetFileNames(seriesIdentifier)

    reader = itk.ImageSeriesReader.New(
        ImageIO=itk.GDCMImageIO.New(),
        FileNames=fileNames,
        ForceOrthogonalDirection=False
    )

    writer = itk.ImageFileWriter.New(
        Input=reader,
        FileName=out_file,
        UseCompression=True
    )
    writer.Update()


def KMeans_segmetation(image, label, apply_filter=False, region=None):
    # Apply KMean segmentation
    kmeans = itk.ScalarImageKmeansImageFilter.New(
        Input=image,
        ImageRegion=region)
    kmeans.AddClassWithInitialMean(0)
    kmeans.AddClassWithInitialMean(70)
    kmeans.AddClassWithInitialMean(150)
    kmeans.AddClassWithInitialMean(250)
    kmeans.Update()

    # Create label map
    kmeans_uc = itk.CastImageFilter[
        itk.Image[itk.SS, 3],
        itk.Image[itk.UC,3]].New(Input=kmeans)
    kmeans_uc.Update()
    label_map = itk.LabelImageToLabelMapFilter.New(Input=kmeans_uc)
    label_map.Update()

    # Extract labled pixels
    obj = itk.LabelMapMaskImageFilter.New(
        Input=label_map, FeatureImage=kmeans, Label=label)
    obj.Update()

    # Apply denoise filter
    result = obj
    if apply_filter:
        filtered_obj = itk.MedianImageFilter.New(Input=obj, Radius=[1, 1, 1])
        filtered_obj.Update()
        result = filtered_obj

    return result


def set_slider(interactor, range, x, y, title, default_value=None, callback=lambda x: x):
    # Set slider properties
    slider = vtk.vtkSliderRepresentation2D()
    slider.SetTitleText(title)
    slider.SetMinimumValue(range[0])
    slider.SetMaximumValue(range[-1])
    slider.SetValue(default_value)
    slider.ShowSliderLabelOn()
    slider.SetSliderWidth(0.03)
    slider.SetSliderLength(0.0001)
    slider.SetEndCapWidth(0)
    slider.SetTitleHeight(0.02)
    slider.SetTubeWidth(0.005)

    # Set the slider position
    slider.GetPoint1Coordinate().SetCoordinateSystemToNormalizedDisplay()
    slider.GetPoint1Coordinate().SetValue(x, y)
    slider.GetPoint2Coordinate().SetCoordinateSystemToNormalizedDisplay()
    slider.GetPoint2Coordinate().SetValue(x + 0.25, y)

    # Add the slider to the UI
    sliderWidget = vtk.vtkSliderWidget()
    sliderWidget.SetInteractor(interactor)
    sliderWidget.SetRepresentation(slider)
    sliderWidget.EnabledOn()

    # Add callback
    def _cb(s, *args):
        slider_representation = s.GetSliderRepresentation()
        value = slider_representation.GetValue()
        callback(value)
    sliderWidget.AddObserver("InteractionEvent", _cb)
    return sliderWidget


def main():
    print("Read DICOM series")

    if not os.path.isfile(image_3d_path):
        DICOMs_to_3Dimage(dicoms_dir, image_3d_path)
        print("Created 3d image from dcm series")

    print("Read 3d image")

    image = itk.imread(image_3d_path)

    print("Image size: ", itk.size(image))
    print("Image type: ", itk.template(image)[-1])

    print("Clipping")

    cropper = itk.ExtractImageFilter.New(Input=image)
    cropper.SetDirectionCollapseToIdentity()
    extraction_region = cropper.GetExtractionRegion()
    extraction_region.SetIndex([235, 420, 1])
    extraction_region.SetSize([410, 290, 27])
    cropper.SetExtractionRegion(extraction_region)
    cropper.Update()

    print("Normalize image")

    img_normalized = itk.RescaleIntensityImageFilter.New(
        Input=cropper,
        OutputMinimum=0,
        OutputMaximum=255
    )

    print("Bluring")

    img_normalized_f = itk.CastImageFilter[itk.Image[itk.SS, 3], itk.Image[itk.F, 3]].New(
        Input=img_normalized)
    diffusion = itk.GradientAnisotropicDiffusionImageFilter.New(
        Input=img_normalized_f, TimeStep=0.0001)

    print("Extract sceleton by KMeans segmentation")

    skeleton_region = diffusion.GetOutput().GetLargestPossibleRegion()
    skeleton_region.SetIndex([235, 420, 1])
    skeleton_region.SetSize([410, 289, 27])
    skeleton = KMeans_segmetation(
        image=diffusion, label=3, apply_filter=0, region=skeleton_region)

    print("Extract lungs by KMeans segmentation")

    lungs_region = diffusion.GetOutput().GetLargestPossibleRegion()
    lungs_region.SetIndex([250, 450,  1])
    lungs_region.SetSize([380, 200, 27])
    lungs = KMeans_segmetation(
                        image=diffusion,
                        label=1,
                        apply_filter=1,
                        region=lungs_region)

    print("Merge sceleton and lungs masks")

    skeleton_lungs = itk.AddImageFilter.New(Input1=lungs, Input2=skeleton)

    print("Set background mask")

    background_mask = itk.BinaryThresholdImageFilter.New(
        Input=skeleton_lungs,
        LowerThreshold=1,
        InsideValue=0,
        OutsideValue=1,
    )

    # Set the background values to be different from the segment values
    print("Set the background values to be different from the segment values")

    background = itk.RescaleIntensityImageFilter.New(
        Input=img_normalized,
        OutputMinimum=4,
        OutputMaximum=255
    )

    # Separate the background squeaks from the general
    print("Separate the background squeaks from the general")

    background_masked = itk.MultiplyImageFilter.New(
        Input1=background,
        Input2=background_mask,
    )

    print("Combine segments and background")

    join = itk.AddImageFilter.New(
        Input1=skeleton_lungs,
        Input2=background_masked,
    )

    itk.imwrite(join, image_3d_seg_path)

    print("VTK visualization")

    image = itk.vtk_image_from_image(itk.imread(image_3d_seg_path))

    mapper = vtk.vtkGPUVolumeRayCastMapper()
    mapper.SetInputData(image)

    volume = vtk.vtkVolume()
    volume.SetMapper(mapper)
    volumeProperty = volume.GetProperty()

    RED = (1, 0, 0)
    VIOLET = (1, 0, 1)
    BLACK = (0, 0, 0)
    WHITE = (1, 1, 1)
    BLUE = (0, 0, 1)

    colorDefault = vtk.vtkColorTransferFunction()
    colorDefault.AddRGBPoint(0, *BLACK)
    colorDefault.AddRGBPoint(1, *RED)
    colorDefault.AddRGBPoint(2, *BLACK)
    colorDefault.AddRGBPoint(3, *VIOLET)
    colorDefault.AddRGBSegment(4, *BLUE, 255, *WHITE)

    opacityDefault = vtk.vtkPiecewiseFunction()
    opacityDefault.AddPoint(0, 0)
    opacityDefault.AddPoint(1, 0.05)
    opacityDefault.AddPoint(2, 0)
    opacityDefault.AddPoint(3, 0)
    opacityDefault.AddSegment(4, 0., 255, 0.01)

    volumeProperty.SetColor(colorDefault)
    volumeProperty.SetScalarOpacity(opacityDefault)
    volumeProperty.SetInterpolationTypeToNearest()

    renderer = vtk.vtkRenderer()
    renderer.AddVolume(volume)

    renderWindow = vtk.vtkRenderWindow()
    renderWindow.AddRenderer(renderer)

    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(renderWindow)

    def cb_opacity_lungs(x): return opacityDefault.AddPoint(1, x/100)
    def cb_opactiy_sceleton(x): return opacityDefault.AddPoint(3, x/100)

    def cb_opacity_background(
        x): return opacityDefault.AddSegment(4, 0., 255, x/100)

    s1 = set_slider(interactor=interactor,
                range=(0, 100),
                x=0.05,
                y=0.1,
                title="Lungs opacity %",
                default_value=20,
                callback=cb_opacity_lungs)
    s2 = set_slider(interactor=interactor,
                range=(0, 100),
                x=0.35,
                y=0.1,
                title="Sceleton opacity %",
                default_value=28,
                callback=cb_opactiy_sceleton)
    s3 = set_slider(interactor=interactor,
                range=(0, 100),
                x=0.65,
                y=0.1,
                title="Background opacity %",
                default_value=1,
                callback=cb_opacity_background)

    renderWindow.SetSize(1000, 500)
    renderWindow.Render()
    interactor.Start()


if __name__ == "__main__":
    main()
