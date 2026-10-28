"""Path-parameterized temporal PNG renderer; not the historical JPG renderer."""
import numpy as np
from PIL import ImageDraw, ImageFont


def render(image, clip, rendering, font_path):
    im = image.copy()
    d = ImageDraw.Draw(im)
    l, t, r, b = clip['roi_bounds_inclusive_xyxy']
    width = rendering['stroke_px']
    # Place strokes outside the region, except where the original region meets the image edge.
    rectangles = [(max(0, l-width), t, max(0, l-1), b),
                  (min(im.width-1, r+1), t, min(im.width-1, r+width), b),
                  (l, max(0, t-width), r, max(0, t-1)),
                  (l, min(im.height-1, b+1), r, min(im.height-1, b+width))]
    if r == im.width-1:
        rectangles[1] = (r-width+1, t, r, b)
    if t == 0:
        rectangles[2] = (l, 0, r, width-1)
    for box in rectangles:
        d.rectangle(box, fill=tuple(rendering['color_rgb']))
    x, y = l-32, t
    d.rectangle((x, y, x+25, y+29), fill=(0, 0, 0))
    font = ImageFont.truetype(str(font_path), 24)
    d.text((x+13, y+15), '1', font=font, fill=(255, 255, 255), anchor='mm')
    raw, marked = np.asarray(image), np.asarray(im)
    assert np.array_equal(raw[t+3:b-3, l+3:r-3], marked[t+3:b-3, l+3:r-3])
    changed = int(np.any(raw != marked, axis=2).sum())
    return im, changed
