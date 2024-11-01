#%%
import cv2
import numpy as np
import os

# Set up the directories for saving frames
if not os.path.exists('frames'):
    os.makedirs('frames')

# Function to detect lanes on an image and overlay them in red
def detect_lanes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur_gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny Edge Detection
    edges = cv2.Canny(blur_gray, 50, 150)
    
    # Define mask for the region of interest
    mask = np.zeros_like(edges)
    ignore_mask_color = 255
    imshape = image.shape
    vertices = np.array([[(0, imshape[0]), (450, 290), (490, 290), (imshape[1], imshape[0])]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Hough Transform Parameters
    rho = 2
    theta = np.pi / 180
    threshold = 15
    min_line_length = 40
    max_line_gap = 20
    line_image = np.copy(image) * 0

    # Hough Line Detection
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    # Draw lines on a blank image
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    
    # Overlay the lines on the original image
    lanes_detected_image = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    return lanes_detected_image

# Function to process video and display original and lane-detected frames side by side
def process_video(video_path):
    # Load video
    cap = cv2.VideoCapture(video_path)
    frame_index = 0
    
    # Define output video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define codec and create VideoWriter object to save the lane-detected video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output_with_lanes.avi', fourcc, fps, (width * 2, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect lanes on each frame
        lanes_frame = detect_lanes(frame)

        # Save the frames
        cv2.imwrite(f'frames/frame_{frame_index:04d}.jpg', lanes_frame)
        frame_index += 1
        
        # Combine original and lanes-detected frames side by side
        combined_frame = np.hstack((frame, lanes_frame))

        # Write the frame to output video
        out.write(combined_frame)

        # Display the combined video frames
        cv2.imshow('Original Video | Lane Detection', combined_frame)
        
        # Break loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release all resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()

# Run the video processing function with the path to the input video
video_path = 'path_to_your_video.mp4'
process_video(video_path)

#%%
import cv2
import numpy as np
import os

# Set up paths for the frames
frames_path = 'path_to_frames_folder'  # Replace with your frames folder path
output_video_path = 'output_with_lanes.avi'

# Function to detect lanes on an image and overlay them in red
def detect_lanes(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur_gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny Edge Detection
    edges = cv2.Canny(blur_gray, 50, 150)
    
    # Define mask for the region of interest
    mask = np.zeros_like(edges)
    ignore_mask_color = 255
    imshape = image.shape
    vertices = np.array([[(0, imshape[0]), (450, 290), (490, 290), (imshape[1], imshape[0])]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Hough Transform Parameters
    rho = 2
    theta = np.pi / 180
    threshold = 15
    min_line_length = 40
    max_line_gap = 20
    line_image = np.copy(image) * 0

    # Hough Line Detection
    lines = cv2.HoughLinesP(masked_edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    # Draw lines on a blank image
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
    
    # Overlay the lines on the original image
    lanes_detected_image = cv2.addWeighted(image, 0.8, line_image, 1, 0)
    return lanes_detected_image

# Get a list of frames sorted by name
frame_files = sorted([f for f in os.listdir(frames_path) if f.endswith('.jpg') or f.endswith('.png')])

# Assuming all frames have the same dimensions
sample_frame = cv2.imread(os.path.join(frames_path, frame_files[0]))
height, width, _ = sample_frame.shape

# Define codec and create VideoWriter object to save the output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_video_path, fourcc, 20, (width * 2, height))

for frame_file in frame_files:
    frame_path = os.path.join(frames_path, frame_file)
    frame = cv2.imread(frame_path)

    # Detect lanes in the current frame
    lanes_frame = detect_lanes(frame)

    # Combine original and lanes-detected frames side by side
    combined_frame = np.hstack((frame, lanes_frame))

    # Write the combined frame to output video
    out.write(combined_frame)

    # Display the combined frames in a window
    cv2.imshow('Original | Lane Detection', combined_frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video writer and close display windows
out.release()
cv2.destroyAllWindows()
