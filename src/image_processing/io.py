import os
import cv2
import image_processing.paths as paths


def read_image(path_to_image, cache=True, process=None):
    filename = os.path.basename(path_to_image)
    read_flags = cv2.IMREAD_COLOR or cv2.IMREAD_ANYDEPTH
    output_file = os.path.join(paths.OUTPUT_FOLDER, filename)
    if cache and os.path.exists(output_file):
        return cv2.imread(output_file, read_flags)

    input_image = cv2.imread(path_to_image, read_flags)
    if process:
        output_image = process(input_image)
    else:
        output_image = input_image
    if cache:
        write_image(os.path.join(paths.OUTPUT_FOLDER, filename), output_image)
    return output_image


def write_image(path_to_image, image):
    cv2.imwrite(path_to_image, image)
