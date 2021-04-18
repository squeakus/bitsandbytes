import cv2
import torch
from torch.autograd import Variable
import torchvision
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image


def main():

    loaded_model = get_model(num_classes=2)
    loaded_model.load_state_dict(torch.load("models/model.pth"))
    image = image_loader("0.jpg")
    print("hello!")
    # Export the model
    torch.onnx.export(
        loaded_model,  # model being run
        image,  # model input (or a tuple for multiple inputs)
        "models/model.onnx",  # where to save the model (can be a file or file-like object)
        export_params=True,  # store the trained parameter weights inside the model file
        do_constant_folding=True,  # whether to execute constant folding for optimization
        opset_version=11,
        input_names=["input"],  # the model's input names
        output_names=["output"],  # the model's output names
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},  # variable length axes
    )


def image_loader(image_name):
    """load image, returns cuda tensor"""
    loader = torchvision.transforms.Compose([transforms.ToTensor()])

    image = Image.open(image_name)
    image = loader(image).float()
    image = Variable(image, requires_grad=True)

    image = image.unsqueeze(0)  # this is for VGG, may not be needed for ResNet
    return image
    return image.cuda()  # assumes that you're using GPU


def get_model(num_classes):
    # load an object detection model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new on
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


if __name__ == "__main__":
    main()