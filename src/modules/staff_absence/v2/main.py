# from __future__ import annotations
# import argparse


# def main() -> None:
#     parser = argparse.ArgumentParser(description="Run the staff absence v2 module.")
#     parser.add_argument("--config", type=str, help="Path to the configuration file.")
#     parser.add_argument("--debug", action="store_true", help="Enable debug output.")
#     args = parser.parse_args()

#     print(f"staff_absence v2 started (config={args.config}, debug={args.debug})")

import cv2
import numpy as np
from .v2_utils import *
from ....utils.json_utils import *

if __name__ == "__main__":
    # main()
    map_pts = load_json("data/annotations/map.json")
    # print("map_pts:", map_pts)

    src_pts = np.array([[120,0],[420,0],[420,560],[300,560]], dtype=np.float32)
    cam_pts = np.array([[117,580], [726,431], [1888, 703], [1837, 842]], dtype=np.float32)
    
    map_cam_pts = HomoGraphyTransform(src_pts, cam_pts, map_pts)
    save_json("data/annotations/map_cam.json", map_cam_pts)

    draw_camera_maps("data/raw/1809_frame.jpg", map_cam_pts)    