import json
import cv2
import os

def main():
    with open('annotations_all.json') as f:
      data = json.load(f)
      categories = data['categories']
      images = data['images']
      annotations = data['annotations']

      
      bbox_list = bboxes_per_image(annotations)

      colors = {"IncompleteFillet": (250, 206, 135),
                "NoEpoxy": (255, 255, 255),
                "ExcessEpoxy": (139, 0, 0),
                "EpoxyOnDie": (255, 0, 0),
                "MissingDie": (0, 0, 255),
                "EpoxySurroundingComponent": (128, 0, 128),
                "DieShift": (0, 165, 255)
                }
      
      for image in images:
        label_image(image, bbox_list, colors)

def label_image(image, bbox_list, colors):

        if image['id'] in bbox_list.keys():
            bboxes = bbox_list[image['id']]['bboxes']
            categories = bbox_list[image['id']]['categories']
            
            if len(image['type']) != len(categories):
                print("category count not matching for:",image['file_name'])
                exit()

            filename = os.path.join("img", image['file_name'])
            image = cv2.imread(filename)
            labelled = image.copy()

            for idx, bbox in enumerate(bboxes):
                start = (bbox[0], bbox[1])
                end = (bbox[0] + bbox[2], bbox[1] + bbox[3])
                color = colors[categories[idx]]
                labelled = cv2.rectangle(labelled, start, end, color, 2)
                cv2.putText(labelled, categories[idx], start, 3, 1, color)

            newname = filename.replace('.jpg', '_labelled.jpg')
            print("saving", newname, "with ", len(bboxes), "bounding boxes")
            cv2.imwrite(newname, labelled)

def bboxes_per_image(annotations):
    bbox_list = {}
    for annotation in annotations:
        image_id = annotation["image_id"]
        category_name = annotation["category_name"]
        bbox = annotation['bbox']
        
        #cast everything to an int
        if any(isinstance(x, float) for x in bbox):
            bbox = [int(x) for x in bbox]

        if image_id in bbox_list.keys():
            bbox_list[image_id]["bboxes"].append(bbox)
            bbox_list[image_id]["categories"].append(category_name)
        else:
            bbox_list[image_id] = {}
            bbox_list[image_id]["bboxes"] = [bbox]
            bbox_list[image_id]["categories"] = [category_name]

    return bbox_list

if __name__ == '__main__':
    main()
