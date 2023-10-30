import xml.etree.ElementTree as ET
from PIL import Image
from PIL import ImageDraw

# color information
colormap = {
    "text": (254, 231, 44),
    "image": (27, 187, 146),
    "headline": (255, 0, 0),
    "text-over-image": (0, 102, 255),
    "headline-over-image": (204, 0, 255),
    "background": (200, 200, 200),
}

background = Image.new('RGBA', (225, 300), colormap["background"])
poly = Image.new('RGBA', (225, 300))
pdraw = ImageDraw.Draw(poly)

tree = ET.parse(r'E:\comprehensive_library\mmgeneration_add\data\MagImage\layoutdata\sample.xml')
root = tree.getroot()

for layout in root.findall('layout'):
    for element in layout.findall('element'):
        label = element.get('label')
        px = [int(i) for i in element.get('polygon_x').split(" ")]
        py = [int(i) for i in element.get('polygon_y').split(" ")]
        pdraw.polygon(list(zip(px, py)),
                      fill=colormap[label], outline=colormap[label])

background.paste(poly, mask=poly)
background.save('sample.png')
