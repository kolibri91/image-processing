## Object detection

We want to apply a object detection algorithm to find people on images in order to mask out areas for a median filter.

## Example

### Apply object detector to full resolution image

Only persons at the lake are detected. Mirroring effects are not detected. Persons in the background (partially covered by the hill) are not detected.
<img src=https://tawiesn.de/leica/zermatt_persons4.png>

### Apply object detector to image part (mirrored)

Persons and their mirror image in the lake is detected
<img src=https://tawiesn.de/leica/zermatt_persons0.png>

### Apply object detector to image part

Also partially covered persons are detected. Please note: the confidence correlates quite well how much of the person is visible in the image. If only about one third of the person is visible (e.g. only the head and a part of the arm) the confidence is about 0.3.

<img src=https://tawiesn.de/leica/zermatt_persons1.png><br>
<img src=https://tawiesn.de/leica/zermatt_persons2.png>

### Detect all persons in full resolution image

Use resolution pyramid and sliding window and detect all persons:

<img src=https://tawiesn.de/leica/zermatt_persons5.jpg>

Due to the pyramid approach some persons might be detected several times in differently resolved images. One could do a maximum suppression of the bounding boxes. On the other side, in the end we are just interested in a pixel mask. So we can skip that step.