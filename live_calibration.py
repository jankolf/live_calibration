import argparse
import numpy as np
import cv2
import sys
import time

def write_text_to_image(image,
                        text, 
                        text_location, 
                        text_size=1,
                        text_color=(0,0,255),
                        line_type = 2):     
    cv2.putText(image,text, 
            text_location, 
            cv2.FONT_HERSHEY_SIMPLEX, 
            text_size,
            text_color,
            line_type)

if __name__ == "__main__":
    
    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['calibrate', 'generate'], help='to calibrate or rectify')
    parser.add_argument('--rows', type=int, default=8, help='Amount of rows (short side of paper)')
    parser.add_argument('--columns', type=int, default=10, help='Amount of columns (long side of paper')
    parser.add_argument('--video-height', type=int, default=720, help="Height of Livestream video")
    parser.add_argument('--video-width', type=int, default=1280, help="Width of Livestream video")
    parser.add_argument('--time-step', type=float, default=3, 
                        help="Min. seconds which need to pass between two captures")
    parser.add_argument('--camera-id', type=int, help='camera_id, default=0', default=0)
    parser.add_argument('--output-file', type=str, help='Output Filename', default="calibration_data")
    args = parser.parse_args()
 
    # Dictionary for Charuco Board
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    board = cv2.aruco.CharucoBoard_create(args.rows,args.columns,0.035,0.0175,dictionary)

    # Generation Mode for the board
    if args.mode == "generate":   
        img = board.draw((2480, 3508)) # DIN A4
        cv2.imwrite("charuco.png", img)
        sys.exit(0)
        
    # Generate the Video Capture
    capture = cv2.VideoCapture(args.camera_id)
    
    if not capture.isOpened():
        capture.release()
        sys.exit("Could not open camera device.")
    
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.video_width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.video_height)

    cv2.namedWindow("Livestream", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Livestream", args.video_width, args.video_height)
    
    text_location = (10, args.video_height - 10)
    text_captured = (10, args.video_height - 40)       
    
 
    try:
        
        last_selected = 0
        charuco_corners_all = []
        charuco_ids_all = []
        
        image_captures = 0
        
        while(True):
            if cv2.waitKey(1) == 27:
                break
            
            ret, frame = capture.read()
    
            if ret == False:
                break
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            marker_corners, marker_ids, marker_rejected  = cv2.aruco.detectMarkers(gray,
                                                                                   dictionary)
            
            if (not marker_corners is None) and len(marker_corners) > 0:
                write_text_to_image(frame, f"#{len(marker_corners)} corners found.", text_location)
                
                ret, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                                                                marker_corners,
                                                                marker_ids,
                                                                frame,
                                                                board)

                if time.time() - last_selected > args.time_step and \
                    not charuco_corners is None and not charuco_ids is None \
                    and len(charuco_corners) > 3:
                    
                    last_selected = time.time()
                    image_captures += 1
                    
                    charuco_corners_all.append(charuco_corners)
                    charuco_ids_all.append(charuco_ids)
                    
            
                cv2.aruco.drawDetectedMarkers(frame, marker_corners, marker_ids)
                cv2.aruco.drawDetectedCornersCharuco(frame, charuco_corners, charuco_ids)
                
            else:
                write_text_to_image(frame, "No charucos found.", text_location)
               
            
            write_text_to_image(frame, "Press Esc. to stop recording.", (10,30))
            write_text_to_image(frame, f"Images captured: {image_captures}", text_captured)
            cv2.imshow('Livestream', frame)


        ret,camera_matrix, dist_coeffs,rvecs,tvecs = cv2.aruco.calibrateCameraCharuco(
                                                charuco_corners_all,
                                                charuco_ids_all,
                                                board,
                                                (args.video_height, args.video_width),
                                                None,None)
        
        np.savez(f"{args.output_file}.npz",
                 camera_matrix=camera_matrix,
                 distortion_coefficients=dist_coeffs,
                 rotation_vectors=rvecs,
                 translation_vectors=tvecs)
        
    except Exception:        
        print("Error occured. Exiting.")  
    finally:       
        capture.release() 
        cv2.destroyAllWindows()
