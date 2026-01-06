import cv2
import numpy as np
import os
import pickle
from pupil_apriltags import Detector
import streamer

# Live undistortion parameters
#CAMERA_ID = 0  # Camera ID (usually 0 for built-in webcam)
URL = "rtsp://thingino:thingino@192.168.0.176:554/ch0"
CAMERA_ID = URL
CALIBRATION_FILE = 'output/calibration_data.pkl'  # Path to calibration data

# Start camera streamer
streamer = streamer.RTSPStreamer(URL)
# Initialize detector for tag36h11 family (common)
at_detector = Detector(families='tag36h11', nthreads=1, quad_sigma=0.0, refine_edges=1)

def live_undistortion():
    """
    Demonstrate live camera undistortion using calibration results.
    """
    # Check if calibration file exists
    if not os.path.exists(CALIBRATION_FILE):
        print(f"Error: Calibration file not found at {CALIBRATION_FILE}")
        print("Please run camera_calibration.py first to generate calibration data.")
        return
    
    # Load calibration data
    with open(CALIBRATION_FILE, 'rb') as f:
        calibration_data = pickle.load(f)
    
    mtx = calibration_data['camera_matrix']
    dist = calibration_data['distortion_coefficients']
    camera_params = [mtx[0,0], mtx[1,1], mtx[0,2], mtx[1,2]] # camera parameters for AprilTag detection from camera matrix
    tag_size = 0.150 # AprilTag size in meters
    
    print("Loaded camera calibration data:")
    print(f"Camera Matrix:\n{mtx}")
    print(f"Distortion Coefficients: {dist.ravel()}")

    frame = None
    while(frame is None):
        frame = streamer.get_latest_frame()
        
    # Get camera resolution
    width = int(frame.shape[1])
    height = int(frame.shape[0])
    
    # Calculate optimal camera matrix
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (width, height), 1, (width, height))
    
    # Create undistortion maps
    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (width, height), 5)
    
    print("Press 'q' to quit, 'd' to toggle distortion correction")
    
    # Flag to toggle distortion correction
    correct_distortion = True
    
    # 1. Define 3D axis points (Origin, X, Y, Z tips)
    # Length is 0.1 units (e.g., 10cm if your tag_size was in meters)
    axis_length = 0.1   # 10 cm
    axis_3d = np.float32([[0,0,0], [axis_length,0,0], [0,axis_length,0], [0,0,axis_length]])
    # 180-degree rotation matrix around X-axis to match coordinate systems
    R_flip = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1]
    ], dtype=np.float32)

    while True:
        # Capture frame
        frame = streamer.get_latest_frame()
        
        if frame is not None:
            if correct_distortion:
                # Apply undistortion
                undistorted = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
                
                # Crop the image (optional)
                x, y, w, h = roi
                undistorted = undistorted[y:y+h, x:x+w]
                
                # Resize to original size for display
                undistorted = cv2.resize(undistorted, (width, height))
                gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
                # Detect tags
                detections = at_detector.detect(gray, 
                                            estimate_tag_pose=True, 
                                            camera_params=camera_params, 
                                            tag_size=tag_size)
                # print text if found tags
                for det in detections:
                    pts = det.corners.astype(int)
                    cv2.polylines(undistorted, [pts], True, (0, 255, 0), 2) # Green box
                    # Draw center
                    cv2.circle(undistorted, (int(det.center[0]), int(det.center[1])), 5, (0, 0, 255), -1) # Red dot
                    
                    # 1. Get the lowest point (maximum Y-value) of the tag's bounding box
                    # r.corners is a 4x2 array: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    # We find the max Y and the average X to center the text horizontally
                    max_y = int(np.max(det.corners[:, 1]))
                    avg_x = int(np.mean(det.corners[:, 0]))

                    # 2. Format the pose_t values (x, y, z)
                    text = f"tvec: {det.pose_t[0][0]:.2f}, {det.pose_t[1][0]:.2f}, {det.pose_t[2][0]:.2f}"
                    err_t = f"err: {det.pose_err:.6f}"

                    # 3. Draw the text slightly below the max_y coordinate
                    # (avg_x - offset) centers the text roughly under the tag
                    cv2.putText(undistorted, text, (avg_x - 100, max_y + 25), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(undistorted, err_t, (avg_x - 30, max_y + 45), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # add an orientation vector from the tag's center
                    # 2. Convert pose_R (3x3 matrix) to rvec (3x1 vector)
                    corrected_R = np.dot(det.pose_R, R_flip)    # align tag coordinate system to camera coordinate system
                    rvec, _ = cv2.Rodrigues(corrected_R)
                    tvec = det.pose_t
                    img_pts, _ = cv2.projectPoints(axis_3d, rvec, tvec, mtx, dist)
                    img_pts = img_pts.astype(int).reshape(-1, 2)
                    origin = tuple(img_pts[0])
                    cv2.line(undistorted, origin, tuple(img_pts[1]), (0, 0, 255), 2) # Red = X
                    cv2.line(undistorted, origin, tuple(img_pts[2]), (0, 255, 0), 2) # Green = Y
                    cv2.line(undistorted, origin, tuple(img_pts[3]), (255, 0, 0), 2) # Blue = Z

                # Add text to indicate undistorted view
                cv2.putText(undistorted, "Undistorted", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # add a small circle to show the camera center
                cv2.circle(undistorted, (int(width/2), int(height/2)), 5, (255, 0, 0), -1) # Blue dot

                # Display the undistorted frame
                cv2.imshow('Camera Feed', undistorted)
            else:
                # Add text to indicate original view
                cv2.putText(frame, "Original", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                # Display the original frame
                cv2.imshow('Camera Feed', frame)
            
        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        
        # 'q' to quit
        if key == ord('q'):
            break
        
        # 'd' to toggle distortion correction
        elif key == ord('d'):
            correct_distortion = not correct_distortion
            print(f"Distortion correction {'ON' if correct_distortion else 'OFF'}")
    
    # Release camera and close windows
    streamer.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    live_undistortion()