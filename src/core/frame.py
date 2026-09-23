import os
import cv2
import numpy as np
from typing import Generator, Tuple, Optional, Union

class Frame:
    def __init__(self, frame):
        self.frame = frame
        self.heigth, self.width = self.frame.shape[:2]
        
    def preprocess(self, target_size: Optional[Tuple[int, int]] = None):
        preprocess_frame = cv2.resize(self.frame, target_size)
        preprocess_frame = cv2.cvtColor(preprocess_frame, cv2.COLOR_BGR2RGB)
        preprocess_frame = preprocess_frame.transpose(2, 0, 1)  # HWC to CHW
        preprocess_frame = np.expand_dims(preprocess_frame, axis=0).astype(np.float32)
        preprocess_frame /= 255.0

        return preprocess_frame

    def letterbox(self, target_size: Optional[Tuple[int, int]] = None):
        '''
            Use letterbox or resize with target
                - Letterbox for high accuracy
                - Resize for low-code and high performance
        '''
        ih, iw = self.height, self.width
        th, tw = target_size[1], target_size[0]
        
        # Scale ratio (smaller scale for fit frame in target one)
        r = min(th / ih, tw / iw)
        
        # New size after scale 
        nw, nh = int(round(iw * r)), int(round(ih * r))
        
        # Resize frame with new size
        resized = cv2.resize(self.frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        
        # Calculate adding padding for target_size (th, tw)
        dw, dh = tw - nw, th - nh
        dw /= 2  
        dh /= 2  
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        # Add (padding) gray (114, 114, 114)
        padded_img = cv2.copyMakeBorder(
            resized, top, bottom, left, right, 
            cv2.BORDER_CONSTANT, value=(114, 114, 114) 
        )
        
        # Save scale infos 
        self.ratio = r
        self.dw = dw
        self.dh = dh
        
        return padded_img


