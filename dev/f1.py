import numpy as np
import cv2

from devlib import get_idx_to_coordinates, \
                   align, \
                    mpFaceMesh, get_masked_frame    

def face_align(frame): 
    """
        frame : RGB converted ndarray
    """
    
    idx_to_coordinates = get_idx_to_coordinates(frame)
    if idx_to_coordinates:
        try:
            frame = get_masked_frame(frame,idx_to_coordinates[0],connection=mpFaceMesh.FACEMESH_FACE_OVAL,thickness=20)
            frame = get_masked_frame(frame,idx_to_coordinates[0],connection=mpFaceMesh.FACEMESH_LIPS,color=(255,0,0),thickness=2)
            frame = align(frame,idx_to_coordinates[0],1)
        except:
            return frame
    return frame
