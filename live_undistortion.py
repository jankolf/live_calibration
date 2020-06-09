# -*- coding: utf-8 -*-
import argparse
import numpy as np
import cv2
import sys
import time

if __name__ == "__main__":
    
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-height', type=int, default=720, help="Height of Livestream video")
    parser.add_argument('--video-width', type=int, default=1280, help="Width of Livestream video")
    parser.add_argument('--camera-id', type=int, help='camera_id, default=0', default=0)
    parser.add_argument('--calibration-file', type=str, help='Calibration file', default="calibration_data")
    args = parser.parse_args()
      
    try:
        calibration_data = np.load(f"{args.calibration_file}.npz")
        cm_matrix = calibration_data["camera_matrix"]
        dist_coeff = calibration_data["distortion_coefficients"]
        del calibration_data
        
        # Generate the Video Capture
        capture = cv2.VideoCapture(args.camera_id)
        
        if not capture.isOpened():
            capture.release()
            sys.exit("Could not open camera device.")
        
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.video_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.video_height)
    
        cv2.namedWindow("Livestream", cv2.WINDOW_NORMAL)
        
        small_size = (args.video_height//2, args.video_width//2)
        
        cv2.resizeWindow("Livestream", small_size[1]*2, small_size[0])
            
        while(True):
            if cv2.waitKey(1) == 27:
                break
            
            ret, frame = capture.read()
    
            if ret == False:
                print("Can't read image from camera. Exiting.")
                break
            
            frame_undistorted = cv2.undistort(frame, cm_matrix, dist_coeff)
            
            res_original = cv2.resize(frame, small_size, interpolation = cv2.INTER_AREA)
            res_undist = cv2.resize(frame_undistorted, small_size, interpolation = cv2.INTER_AREA)

            cv2.imshow('Livestream', np.hstack((res_original, res_undist)))

    except Exception:        
        print("Error occured. Exiting.")  
    finally:       
        capture.release() 
        cv2.destroyAllWindows()
