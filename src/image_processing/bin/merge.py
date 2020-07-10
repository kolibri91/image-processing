import argparse
import os
import sys
from functools import partial

import cv2
import image_processing.paths as paths
import image_processing.pipeline as pipeline
import numpy as np
from image_processing.io import read_image, write_image
from matplotlib import pyplot as plt


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", help="Files to process")
    parser.add_argument("--out", help="Path for result image")
    parser.add_argument("--preview", action="store_true", help="Process 4x downsamples image")
    parser.add_argument("--show-matches", action="store_true", help="Show matches")
    parser.add_argument("--show-result", action="store_true", help="Show result")
    return parser


def parse_command_line_args(args):
    parsed_args = get_parser().parse_args(args)
    return vars(parsed_args)


def main(**kwargs):
    if kwargs["preview"]:
        resize = pipeline.Pipeline()
        resize.add(pipeline.Resize((1/4, 1/4)))
        images = list(map(partial(read_image, cache=True, process=resize), kwargs["files"]))
    else:
        images = list(map(partial(read_image, cache=False), kwargs["files"]))
    image_size = (images[0].shape[1], images[0].shape[0])
    datatype = images[0].dtype
    images_8bit = list(map(np.uint8, images)) if datatype != np.uint8 else images

    feature_detector = cv2.ORB_create()
    feature_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    (keypoints, description) = list(zip(*map(partial(feature_detector.detectAndCompute, mask=None), images_8bit)))
    matches = []
    for desc in description[1:]:
        sorted_matches = sorted(
            feature_matcher.match(description[0], desc),
            key=lambda x: x.distance
        )
        matches.append(sorted_matches)

    if kwargs["show_matches"]:
        for (idx, match) in enumerate(matches):
            img_matched = cv2.drawMatches(images_8bit[0], keypoints[0], images_8bit[idx + 1], keypoints[idx + 1], match[:10], None)
            plt.imshow(img_matched)
            plt.show()

    homographies = [np.eye(3)]
    for (idx, match) in enumerate(matches):
        dst_pts = np.float32([keypoints[0][m.queryIdx].pt for m in match]).reshape((-1, 1, 2))
        src_pts = np.float32([keypoints[idx + 1][m.trainIdx].pt for m in match]).reshape((-1, 1, 2))
        homography, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        homographies.append(homography)

    warp = partial(cv2.warpPerspective, dsize=image_size)
    warped_images = np.array(list(map(lambda x: warp(*x), zip(images, homographies))))
    result_image = np.median(warped_images, axis=0).astype(datatype)

    if kwargs["show_result"]:
        plt.imshow(result_image[..., ::-1])
        plt.show()

    if kwargs["out"]:
        write_image(kwargs["out"], result_image)
    else:
        _, extension = os.path.splitext(kwargs["files"][0])
        write_image(os.path.join(paths.RESULT_FOLDER, "merged" + extension), result_image)
    return 0


if __name__ == "__main__":
    arguments = sys.argv[1:]
    # arguments = [
    #     os.path.join(paths.INPUT_FOLDER, "Zermatt_20200622-07.tif"),
    #     os.path.join(paths.INPUT_FOLDER, "Zermatt_20200622-08.tif"),
    #     os.path.join(paths.INPUT_FOLDER, "Zermatt_20200622-09.tif"),
    #     os.path.join(paths.INPUT_FOLDER, "Zermatt_20200622-10.tif"),
    #     os.path.join(paths.INPUT_FOLDER, "Zermatt_20200622-11.tif"),
    #     os.path.join(paths.INPUT_FOLDER, "Zermatt_20200622-12.tif"),
    #     os.path.join(paths.INPUT_FOLDER, "Zermatt_20200622-13.tif"),
    #     os.path.join(paths.INPUT_FOLDER, "Zermatt_20200622-14.tif"),
    #     os.path.join(paths.INPUT_FOLDER, "Zermatt_20200622-15.tif"),
    #     os.path.join(paths.INPUT_FOLDER, "Zermatt_20200622-16.tif"),
    #     os.path.join(paths.INPUT_FOLDER, "Zermatt_20200622-17.tif"),
    #     os.path.join(paths.INPUT_FOLDER, "Zermatt_20200622-18.tif"),
    #     os.path.join(paths.INPUT_FOLDER, "Zermatt_20200622-19.tif"),
    #     "--preview",
    #     "--show-result",
    # ]
    sys.exit(main(**parse_command_line_args(arguments)))
