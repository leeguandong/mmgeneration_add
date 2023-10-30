'''
@Time    : 2022/7/15 9:33
@Author  : leeguandon@gmail.com
'''
from PIL import Image, ImageDraw


def convert_xywh_to_ltrb(bbox):
    xc, yc, w, h = bbox
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return [x1, y1, x2, y2]


def convert_layout_to_image(boxes, labels, colors, canvas_size):
    H, W = canvas_size
    img = Image.new('RGB', (int(W), int(H)), color=(255, 255, 255))
    draw = ImageDraw.Draw(img, 'RGBA')

    # draw from larger boxes
    area = [b[2] * b[3] for b in boxes]
    indices = sorted(range(len(area)),
                     key=lambda i: area[i],
                     reverse=True)

    for i in indices:
        bbox, color = boxes[i], colors[labels[i]]
        c_fill = color + (100,)
        x1, y1, x2, y2 = convert_xywh_to_ltrb(bbox)
        x1, x2 = x1 * (W - 1), x2 * (W - 1)
        y1, y2 = y1 * (H - 1), y2 * (H - 1)
        draw.rectangle([x1, y1, x2, y2],
                       outline=color,
                       fill=c_fill)
    return img
