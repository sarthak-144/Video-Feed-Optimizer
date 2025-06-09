import cv2
import time
import os
import numpy as np

def apply_clahe_video_denoised(input_path, output_path, denoise_kernel_size=5):
    """
    Applies CLAHE, Denoising, and Sharpening to a video.

    Args:
        input_path (str): Path to the input video file.
        output_path (str): Path to save the processed video file.
        denoise_kernel_size (int): Kernel size for median blur (e.g., 3, 5). 
                                   Must be an odd number. Set to 0 or 1 to disable.
    """
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(input_path)

    # Check if video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video file: {input_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) # Get total frames for progress

    # Try different codecs if mp4v doesn't work, e.g., 'XVID' for .avi
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Check if VideoWriter was initialized successfully
    if not out.isOpened():
        print(f"Error: Could not open video writer for path: {output_path}")
        print("Ensure you have necessary codecs installed (e.g., ffmpeg) and the path is writable.")
        cap.release()
        return

    # CLAHE object
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    
    # Sharpening kernel
    # You can make this less aggressive if needed, e.g., by reducing the center value
    # or by blending the sharpened frame with the original.
    # kernel_sharpen = np.array([[-1, -1, -1], 
    #                            [-1, 9, -1],
    #                            [-1, -1, -1]])
    kernel_sharpen = np.array([[0, -1, 0], 
                               [-1, 5, -1], # Center value controls sharpness amount
                               [0, -1, 0]])
    # kernel_sharpen = np.array([[0, 0, 0], 
    #                            [0, 1, 0],
    #                            [0, 0, 0]])

    print(f"Processing video: {input_path}")
    print(f"Outputting to: {output_path}")
    print(f"FPS: {fps}, Dimensions: {width}x{height}, Total Frames: {total_frames if total_frames > 0 else 'N/A'}")
    if denoise_kernel_size > 1 and denoise_kernel_size % 2 == 1:
        print(f"Denoising enabled with gaussian blur kernel size: {denoise_kernel_size}")
    else:
        print("Denoising disabled or invalid kernel size.")

    frame_count = 0
    overall_start_time = time.time()

    while True:
        loop_start_time = time.time() # Start time for this iteration

        ret, frame = cap.read()
        if not ret:
            print("\nFinished processing all frames or encountered an error reading frame.")
            break

        frame_count += 1
        
        # Convert to YCrCb color space (Luminance, Chrominance)
        img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        
        # Separate the Y channel (luminance)
        y_channel = img_yuv[:, :, 0]
        cr_channel = img_yuv[:, :, 1]
        cb_channel = img_yuv[:, :, 2]

        # 1. Apply CLAHE to the Y channel
        y_channel_clahe = clahe.apply(y_channel)
        
        # 2. Denoise the Y channel (after CLAHE, before sharpening)
        y_channel_processed = y_channel_clahe
        if denoise_kernel_size > 1 and denoise_kernel_size % 2 == 1: # Ensure kernel is odd and > 1
            # y_channel_denoised = cv2.GaussianBlur(y_channel_clahe, (denoise_kernel_size, denoise_kernel_size), 0)
            y_channel_denoised = cv2.bilateralFilter(y_channel_clahe,9,75,75)
            y_channel_processed = y_channel_denoised
        else:
            y_channel_processed = y_channel_clahe # Skip denoising

        # 3. Sharpen the processed Y channel
        y_channel_sharpened = cv2.filter2D(y_channel_processed, -1, kernel_sharpen)
        
        # Merge the processed Y channel back with original Cr and Cb channels
        processed_yuv = cv2.merge([y_channel_sharpened, cr_channel, cb_channel])
        
        # Convert back to BGR color space
        processed_frame = cv2.cvtColor(processed_yuv, cv2.COLOR_YCrCb2BGR)

        out.write(processed_frame)
        
        loop_end_time = time.time() # End time for this iteration
        time_taken_for_frame = loop_end_time - loop_start_time
        
        # Print progress and time per frame
        if total_frames > 0:
            print(f"\rProcessing frame {frame_count}/{total_frames} - Time for frame: {time_taken_for_frame:.4f}s", end="")
        else:
            print(f"\rProcessing frame {frame_count} - Time for frame: {time_taken_for_frame:.4f}s", end="")

    overall_end_time = time.time()
    total_processing_time = overall_end_time - overall_start_time
    print(f"\nTotal video processing time: {total_processing_time:.2f} seconds.")
    if frame_count > 0:
        print(f"Average time per frame: {total_processing_time / frame_count:.4f} seconds.")

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows() # Good practice
    print("Video processing complete. Resources released.")

if __name__ == '__main__':
    # IMPORTANT: Replace with your actual video file path
    input_vid = 'Video Feed optimizer/WhatsApp Video 2025-06-03 at 12.39.50_fce95b5c.mp4' # Example, replace this
    # Ensure this output path is writable and the directory exists or can be created
    output_vid = 'output/clahe_denoised_sharpened_video.mp4'
    
    # Check if input file exists before starting
    if not os.path.exists(input_vid):
        print(f"Error: Input video file not found at '{input_vid}'. Please check the path.")
    else:
        # Apply CLAHE, Denoising (medianBlur with kernel 5), and Sharpening
        # To disable denoising, set denoise_kernel_size to 0 or 1.
        # For less denoising, try denoise_kernel_size=3.
        apply_clahe_video_denoised(input_vid, output_vid, denoise_kernel_size=5)
