import numpy as np
import torch
import torch.utils.data
from PIL import Image, ImageDraw
import pandas as pd
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from engine import train_one_epoch, evaluate
import utils
import os
import glob
import transforms as T


def main():
    data_folder = "/home/jonathan/tmp/labelled"
    if not os.path.exists("models"):
        print("creating model dir")
        os.makedirs("models")

    csv_file = os.path.join(data_folder, "annotations.csv")
    test(data_folder, csv_file)
    # train(data_folder, csv_file)
    # classify(data_folder)


def classify(data_folder):
    loaded_model = get_model(num_classes=2)
    loaded_model.load_state_dict(torch.load("models/model.pth"))
    dataset_classify = ObjectDataset(data_folder=data_folder, csv_file=None, transforms=get_transform(train=False))
    for idx in range(len(dataset_classify)):
        img, _ = dataset_classify[idx]
        label_boxes = np.array(dataset_classify[idx][1]["boxes"])
        # put the model in evaluation mode
        loaded_model.eval()
        with torch.no_grad():
            prediction = loaded_model([img])
            image = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
            draw = ImageDraw.Draw(image)

        for element in range(len(prediction[0]["boxes"])):
            boxes = prediction[0]["boxes"][element].cpu().numpy()
            score = np.round(prediction[0]["scores"][element].cpu().numpy(), decimals=4)
            if score > 0.8:
                draw.rectangle([(boxes[0], boxes[1]), (boxes[2], boxes[3])], outline="red", width=3)
                c_x = (boxes[0] + boxes[2]) / 2
                c_y = (boxes[1] + boxes[3]) / 2
                print("box_center", c_x, c_y)
                draw.ellipse((c_x - 1, c_y - 1, c_x + 1, c_y + 1), fill=(0, 255, 0))
                draw.text((boxes[0], boxes[1]), text=str(score))
        depth, height, width = img.size()
        c_x, c_y = width / 2, height / 2
        print("actual_center", c_x, c_y)

        draw.ellipse((c_x - 1, c_y - 1, c_x + 1, c_y + 1), fill=(255, 0, 0))

        image.save(
            str(idx) + ".jpg",
        )


def test(data_folder, csv_file):
    loaded_model = get_model(num_classes=2)
    loaded_model.load_state_dict(torch.load("models/model.pth"))
    dataset_test = ObjectDataset(data_folder=data_folder, csv_file=csv_file, transforms=get_transform(train=False))
    # split the dataset in train and test set
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset_test)).tolist()
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-10:])

    for idx in range(len(dataset_test)):
        img, _ = dataset_test[idx]
        label_boxes = np.array(dataset_test[idx][1]["boxes"])
        # put the model in evaluation mode
        loaded_model.eval()
        with torch.no_grad():
            prediction = loaded_model([img])
            image = Image.fromarray(img.mul(255).permute(1, 2, 0).byte().numpy())
            draw = ImageDraw.Draw(image)

        # draw groundtruth
        for elem in range(len(label_boxes)):
            draw.rectangle(
                [
                    (label_boxes[elem][0], label_boxes[elem][1]),
                    (label_boxes[elem][2], label_boxes[elem][3]),
                ],
                outline="green",
                width=3,
            )
        for element in range(len(prediction[0]["boxes"])):
            boxes = prediction[0]["boxes"][element].cpu().numpy()
            score = np.round(prediction[0]["scores"][element].cpu().numpy(), decimals=4)
            if score > 0.8:
                draw.rectangle([(boxes[0], boxes[1]), (boxes[2], boxes[3])], outline="red", width=3)
                draw.text((boxes[0], boxes[1]), text=str(score))
        image.save(
            str(idx) + ".jpg",
        )


def train(data_folder, csv_file, train_test_split=0.1):
    dataset = ObjectDataset(data_folder=data_folder, csv_file=csv_file, transforms=get_transform(train=True))
    dataset_test = ObjectDataset(data_folder=data_folder, csv_file=csv_file, transforms=get_transform(train=False))
    # split the dataset in train and test set
    torch.manual_seed(1)
    indices = torch.randperm(len(dataset)).tolist()
    split = int(len(dataset) * train_test_split)
    dataset = torch.utils.data.Subset(dataset, indices[:-split])
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-split:])
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4, collate_fn=utils.collate_fn
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        collate_fn=utils.collate_fn,
    )
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(
        f"{len(indices)} labelled images, {len(dataset)} are training and {len(dataset_test)} testing, running on {device}"
    )

    # device = torch.device('cpu')
    # our dataset has two classes only - zero and background
    num_classes = 2
    # get the model using our helper function
    model = get_model(num_classes)
    # move model to the right device
    model.to(device)
    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler which decreases the learning rate by # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # let's train it for 10 epochs
    num_epochs = 10
    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
    # evaluate on the test dataset
    evaluate(model, data_loader_test, device=device)

    torch.save(model.state_dict(), "models/model.pth")


def get_model(num_classes):
    # load an object detection model pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new on
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
        transforms.append(T.RandomRotate(0.5))
    return T.Compose(transforms)


class ObjectDataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, csv_file, transforms=None):
        self.data_folder = data_folder
        if csv_file is None:
            self.training = False
        else:
            self.csv_file = csv_file
            self.training = True
        self.transforms = transforms
        self.annotations = pd.read_csv(csv_file)
        self.imgs = sorted(self.annotations.filename.tolist())

    def __getitem__(self, idx):
        # load images and bounding boxes
        img_path = os.path.join(self.data_folder, self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        if self.training:
            box_list = self.parse_one_annot(self.imgs[idx])
            boxes = torch.as_tensor(box_list, dtype=torch.float32)
            num_objs = len(box_list)
            # there is only one class
            labels = torch.ones((num_objs,), dtype=torch.int64)
            image_id = torch.tensor([idx])
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            # suppose all instances are not crowd
            iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            target["image_id"] = image_id
            target["area"] = area
            target["iscrowd"] = iscrowd
        else:
            target = {}
            target["boxes"] = None
            target["labels"] = None
            target["image_id"] = None
            target["area"] = None
            target["iscrowd"] = None

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def parse_one_annot(self, filename):
        data = self.annotations
        boxes_array = data[data["filename"] == filename][["xmin", "ymin", "xmax", "ymax"]].values
        return boxes_array

    def __len__(self):
        return len(self.imgs)


def find(regex, folder="./"):
    found = []
    for fullname in glob.iglob(folder + "/**/" + regex, recursive=True):
        filename = fullname.split(os.sep)[-1]
        found.append(filename)
    return found


if __name__ == "__main__":
    main()
