from cubemos.core.nativewrapper import CM_TargetComputeDevice
from cubemos.core.nativewrapper import initialise_logging, CM_LogLevel
from cubemos.skeleton_tracking.nativewrapper import Api, SkeletonKeypoints
import cv2
import os
import platform
import pyrealsense2 as rs
import numpy as np

keypoint_ids = [
    (1, 2),
    (1, 5),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (1, 0),
    (0, 14),
    (14, 16),
    (0, 15),
    (15, 17),
]


def default_license_dir():
    if platform.system() == "Windows":
        return os.path.join(os.environ["LOCALAPPDATA"], "Cubemos", "SkeletonTracking", "license")
    elif platform.system() == "Linux":
        return os.path.join(os.environ["HOME"], ".cubemos", "skeleton_tracking", "license")
    else:
        raise Exception("{} is not supported".format(platform.system()))


def check_license_and_variables_exist():
    license_path = os.path.join(default_license_dir(), "cubemos_license.json")
    if not os.path.isfile(license_path):
        raise Exception(
            "The license file has not been found at location \"" +
            default_license_dir() + "\". "
            "Please have a look at the Getting Started Guide on how to "
            "use the post-installation script to generate the license file")
    if "CUBEMOS_SKEL_SDK" not in os.environ:
        raise Exception(
            "The environment Variable \"CUBEMOS_SKEL_SDK\" is not set. "
            "Please check the troubleshooting section in the Getting "
            "Started Guide to resolve this issue." 
        )


def get_valid_limbs(keypoint_ids, skeleton, confidence_threshold):
    limbs = [
        (tuple(map(int, skeleton.joints[i])), tuple(map(int, skeleton.joints[v])))
        for (i, v) in keypoint_ids
        if skeleton.confidences[i] >= confidence_threshold
        and skeleton.confidences[v] >= confidence_threshold
    ]
    valid_limbs = [
        limb
        for limb in limbs
        if limb[0][0] >= 0 and limb[0][1] >= 0 and limb[1][0] >= 0 and limb[1][1] >= 0
    ]
    return valid_limbs


def render_result(skeletons, img, confidence_threshold):
    skeleton_color = (0, 255, 255)
    for index, skeleton in enumerate(skeletons):
        limbs = get_valid_limbs(keypoint_ids, skeleton, confidence_threshold)
        for limb in limbs:
            cv2.line(
                img, limb[0], limb[1], skeleton_color, thickness=5, lineType=cv2.LINE_AA
            )


def main():
    try:
        check_license_and_variables_exist()   
        sdk_path = os.environ["CUBEMOS_SKEL_SDK"]
        #initialize the api with a valid license key in default_license_dir()
        api = Api(default_license_dir())
        model_path = os.path.join(
            sdk_path, "models", "skeleton-tracking", "fp32", "skeleton-tracking.cubemos"
        )
        api.load_model(CM_TargetComputeDevice.CM_CPU, model_path)
    except Exception as ex:
        print("Exception occured: \"{}\"".format(ex))

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    pipeline.start(config)

    c = rs.colorizer()
    align_to = rs.stream.color
    align = rs.align(align_to)
    while True:
        frames = pipeline.wait_for_frames()
        # Align the depth frame to color frame
        aligned_frames = align.process(frames)

        color = aligned_frames.get_color_frame()
        depth = aligned_frames.get_depth_frame()
        if not depth: continue        

        color_data = color.as_frame().get_data()
        color_image = np.asanyarray(color_data)

        skeletons = api.estimate_keypoints(color_image, 192)

        # perform inference again to demonstrate tracking functionality.
        # usually you would estimate the keypoints on another image and then
        # update the tracking id
        #new_skeletons = api.estimate_keypoints(img, 192)
        #new_skeletons = api.update_tracking_id(skeletons, new_skeletons)

        render_result(skeletons, color_image, confidence_threshold=0.5)

        depth_colormap = c.colorize(depth)
        depth_data = depth_colormap.as_frame().get_data()
        depth_image = np.asanyarray(depth_data)
        depth_image = cv2.cvtColor(depth_image, cv2.COLOR_RGB2BGR)

        added_image = cv2.addWeighted(color_image,0.5,depth_image,0.5,0)

        cv2.imshow('color', color_image)
        cv2.imshow('depth', depth_image)
        cv2.imshow('overlay', added_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    return

if __name__ == '__main__':
    main()
