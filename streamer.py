import cv2
import threading
import time

class RTSPStreamer:
    def __init__(self, rtsp_url, retry_interval=5):
        self.rtsp_url = rtsp_url
        self.retry_interval = retry_interval  # Seconds to wait between reconnect attempts
        self.cap = None
        self.frame = None
        self.stopped = False
        self.is_connected = False
        
        # Start the background thread
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def _connect(self):
        """Internal method to initialize the video capture."""
        if self.cap is not None:
            self.cap.release()
            
        print(f"Attempting to connect to {self.rtsp_url}...")
        self.cap = cv2.VideoCapture(self.rtsp_url)
        
        # Optimization: Set buffer size to 1 to reduce latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        if self.cap.isOpened():
            self.is_connected = True
            print("Successfully connected.")
        else:
            self.is_connected = False
            print("Connection failed.")

    def update(self):
        """Continuously grabs frames and handles reconnection."""
        self._connect()
        
        while not self.stopped:
            if not self.is_connected:
                time.sleep(self.retry_interval)
                self._connect()
                continue

            ret, frame = self.cap.read()
            
            if ret:
                self.frame = frame
            else:
                print("Lost stream. Retrying...")
                self.is_connected = False
                # Brief sleep to prevent high CPU usage during disconnects
                time.sleep(0.1)

    def get_latest_frame(self):
        """Returns the most recent frame. Check for None if stream is down."""
        return self.frame

    def stop(self):
        """Stops the thread and releases resources."""
        self.stopped = True
        if self.thread.is_alive():
            self.thread.join()
        if self.cap:
            self.cap.release()