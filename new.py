import cv2
import numpy as np
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import QTimer, Qt, QCoreApplication

class VideoPlayer(QMainWindow):
    def __init__(self, parent=None):
        super(VideoPlayer, self).__init__(parent)

        # --- Window Setup ---
        self.setWindowTitle("Webcam with Optimized Contrast Equalization")
        # Main window size, allowing some space for the fixed video label
        self.setGeometry(100, 100, 680, 520) # x, y, width, height

        # --- Central Widget and Layout ---
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        layout = QVBoxLayout(self.central_widget)
        layout.setAlignment(Qt.AlignCenter) # Align content in the center

        # --- Video Display Label ---
        self.video_label = QLabel(self)
        # Fixed size for the video display area
        self.video_label_width = 1920
        self.video_label_height = 1080
        self.video_label.setFixedSize(self.video_label_width, self.video_label_height)
        self.video_label.setAlignment(Qt.AlignCenter)
        # Add widget to layout, it will be centered due to layout.setAlignment and fixed size
        layout.addWidget(self.video_label)
        self.video_label.setStyleSheet(
            "border: 1px solid #555; "
            "border-radius: 8px; "
            "background-color: #282828;" # Darker background for the video area
        )

        # --- OpenCV Video Capture ---
        self.cap = cv2.VideoCapture(0) # 0 for default webcam
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            self.video_label.setText(
                "Error: Could not open webcam.\n"
                "Please check camera permissions and connections.\n"
                "Ensure no other application is using the camera."
            )
            self.video_label.setStyleSheet(
                "color: red; font-size: 14px; font-weight: bold; "
                "border: 2px solid red; border-radius: 8px; padding: 20px; "
                "background-color: #400000;"
            )
            self.timer = None # No timer if camera fails
            return

        # --- QTimer for Frame Updates ---
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        # Start timer - 30ms for ~33fps, 40ms for 25fps.
        # Adjust if lag is still present.
        self.timer.start(35) # Changed to 40ms (25 FPS) for potentially smoother experience
        
        # --- Target processing width (can be adjusted) ---
        # Smaller values reduce lag but also detail in processing.
        self.target_processing_width = 480 # e.g., 320, 480, or 640

        print("Webcam initialized. Starting video stream...")

    def update_frame(self):
        """Reads a frame from the webcam, processes it, and updates the QLabel."""
        if not self.cap or not self.cap.isOpened():
            return

        ret, frame = self.cap.read()
        if not ret:
            print("Error: Can't receive frame (stream end?).")
            # Optionally, display a message or stop the timer
            # self.timer.stop() # Uncomment to stop on error
            # self.video_label.setText("Error: Could not read frame from webcam.")
            # self.video_label.setStyleSheet("color: orange; font-size: 16px;")
            return

        # --- Resize for processing (to reduce lag) ---
        original_h, original_w, _ = frame.shape
        frame_to_process = frame # Default to original frame

        if original_w > self.target_processing_width:
            aspect_ratio = original_h / original_w
            processing_h = int(self.target_processing_width * aspect_ratio)
            dim = (self.target_processing_width, processing_h)
            # Use INTER_AREA for shrinking, it's generally good to avoid moire patterns
            frame_to_process = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        # else: frame is smaller or equal to target_processing_width, process as is

        # --- Image Processing: Histogram Equalization on Y channel of YCrCb ---
        processed_frame = self.apply_contrast_equalization(frame_to_process)

        # --- Convert OpenCV image (BGR) to QImage ---
        # The processed_frame might be smaller than the original webcam frame
        rgb_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        h_proc, w_proc, ch_proc = rgb_image.shape
        bytes_per_line = ch_proc * w_proc
        qt_image = QImage(rgb_image.data, w_proc, h_proc, bytes_per_line, QImage.Format_RGB888)

        # --- Display QImage in QLabel ---
        # Scale the (potentially smaller) processed image up to the fixed label size
        pixmap = QPixmap.fromImage(qt_image)
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label_width,  # Use fixed width
            self.video_label_height, # Use fixed height
            Qt.KeepAspectRatio,      # Maintain aspect ratio
            Qt.SmoothTransformation  # Use smooth filter for scaling
        ))

    def apply_contrast_equalization(self, frame_bgr):
        """
        Applies histogram equalization to the Y (luma) channel of a BGR image
        by converting to YCrCb color space.
        """
        try:
            # Convert BGR to YCrCb
            ycrcb_image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
            y, cr, cb = cv2.split(ycrcb_image)

            # Apply histogram equalization to the Y channel
            y_equalized = cv2.equalizeHist(y)

            # Merge the equalized Y channel back with Cr and Cb
            ycrcb_equalized_image = cv2.merge([y_equalized, cr, cb])

            # Convert back to BGR
            bgr_equalized_image = cv2.cvtColor(ycrcb_equalized_image, cv2.COLOR_YCrCb2BGR)
            return bgr_equalized_image
        except cv2.error as e:
            print(f"OpenCV error during YCrCb equalization: {e}")
            # Return original frame if equalization fails
            return frame_bgr
        except Exception as e:
            print(f"Unexpected error during YCrCb equalization: {e}")
            return frame_bgr


    def closeEvent(self, event):
        """Handles the window close event."""
        print("Closing application...")
        if self.timer:
            self.timer.stop()
        if hasattr(self, 'cap') and self.cap and self.cap.isOpened():
            self.cap.release() # Release the webcam
        event.accept()

def main():
    app = QCoreApplication.instance()
    if app is None:
        app = QApplication(sys.argv)

    player = VideoPlayer()
    if player.cap and player.cap.isOpened(): # Only show if camera was successfully opened
        player.show()
        sys.exit(app.exec_())
    else:
        # If camera failed to open in __init__, VideoPlayer might not be fully usable.
        # QApplication might still be needed for the error message label if it's complex.
        # For simplicity, if camera fails, we might just exit or show a simpler error.
        # The current code shows the error in the label of the main window.
        # If the window is shown even on camera error:
        if player.timer is None: # Indicates camera init failed
             player.show() # Show window with error message
             sys.exit(app.exec_()) # Keep app running to show error
        # else:
        #     print("Exiting due to camera initialization failure.")
        #     sys.exit(-1)


if __name__ == '__main__':
    main()
