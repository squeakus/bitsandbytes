import random
import torch

from torchvision.transforms import functional as F


def _flip_coco_person_keypoints(kps, width):
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]
    flipped_data[..., 0] = width - flipped_data[..., 0]
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


def _rotate_bboxes(bboxes, w, h, rotation):
    rotated = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox

        if rotation == 3:  # clockwise
            x3 = (h - 1) - y1
            y3 = x1
            x4 = (h - 1) - y2
            y4 = x2
        elif rotation == 1:  # counterclockwise
            x3 = y1
            y3 = (w - 1) - x1
            x4 = y2
            y4 = (w - 1) - x2

        xmin = min(x3, x4)
        ymin = min(y3, y4)
        xmax = max(x3, x4)
        ymax = max(y3, y4)
        rotated.append([xmin, ymin, xmax, ymax])
    return rotated


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints

        return image, target


class RandomRotate(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        # this augmentation doesn't handle masks or person keypoints
        if random.random() < self.prob:
            rotation = random.choice((1, 3))  # CCW, CW
            height, width = image.shape[-2:]
            image = torch.rot90(image, rotation, [1, 2])

            box_list = _rotate_bboxes(target["boxes"].tolist(), width, height, rotation)
            target["boxes"] = torch.as_tensor(box_list, dtype=torch.float32)

        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target
