from paddleocr import PaddleOCR,draw_ocr
from PIL import Image


ocr = PaddleOCR(use_angle_cls=True, lang='en')
img_path = './test2.jpg'
result = ocr.ocr(img_path, cls=True)
image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]

im_show = draw_ocr(image, boxes, txts, scores, font_path='./fonts/simfang.ttf')
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')
