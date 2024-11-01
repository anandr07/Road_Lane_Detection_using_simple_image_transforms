#%%
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
from moviepy.editor import VideoFileClip
import collections
import math

# Ensure output directory exists
output_dir = "test_videos_output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Function to convert an image to grayscale
def grayscale(img):
    """
    Converts an RGB image to grayscale.
    
    Args:
    img (numpy.ndarray): The input RGB image.

    Returns:
    numpy.ndarray: Grayscale version of the input image.
    """
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Function to detect edges using Canny edge detection
def canny(img, low_threshold=50, high_threshold=150):
    """
    Applies Canny edge detection to an image.
    
    Args:
    img (numpy.ndarray): Grayscale or single-channel image.
    low_threshold (int): Lower bound for the hysteresis thresholding.
    high_threshold (int): Upper bound for the hysteresis thresholding.

    Returns:
    numpy.ndarray: Image with edges detected.
    """
    return cv2.Canny(img, low_threshold, high_threshold)

# Function to apply Gaussian blur to an image
def gaussian_blur(img, kernel_size=15):
    """
    Reduces image noise using a Gaussian blur.
    
    Args:
    img (numpy.ndarray): Input image.
    kernel_size (int): Size of the Gaussian kernel. Must be an odd number.

    Returns:
    numpy.ndarray: Blurred image.
    """
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

# Function to mask the region of interest in an image
def region_of_interest(img, vertices):
    """
    Applies a mask to keep only the region of the image defined by vertices.
    
    Args:
    img (numpy.ndarray): Input image.
    vertices (numpy.ndarray): Array of vertices defining the polygonal mask.

    Returns:
    numpy.ndarray: Masked image with only the region of interest retained.
    """
    mask = np.zeros_like(img)
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    return cv2.bitwise_and(img, mask)

# Function to draw lines on an image
def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    """
    Draws lines on an image.
    
    Args:
    img (numpy.ndarray): Image to draw lines on.
    lines (list): List of lines, where each line is represented by (x1, y1, x2, y2).
    color (list): Color of the lines in RGB.
    thickness (int): Thickness of the lines.
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

# Function to detect lines using Hough Line Transformation
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    Uses Hough Transformation to detect lines in an image.
    
    Args:
    img (numpy.ndarray): Edge-detected image.
    rho (float): Distance resolution in pixels.
    theta (float): Angle resolution in radians.
    threshold (int): Minimum number of intersections needed to detect a line.
    min_line_len (int): Minimum number of pixels in a line.
    max_line_gap (int): Maximum gap between pixels to connect them as a line.

    Returns:
    numpy.ndarray: Array of detected lines.
    """
    return cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)

# Function to overlay one image on top of another with specific weights
def weighted_img(img, initial_img, α=0.8, β=1., γ=0.):
    """
    Combines two images using weighted addition.
    
    Args:
    img (numpy.ndarray): Image with lines drawn on it.
    initial_img (numpy.ndarray): Original image before line detection.
    α (float): Weight for the initial image.
    β (float): Weight for the image with lines.
    γ (float): Scalar added to each pixel in the output.

    Returns:
    numpy.ndarray: Combined image.
    """
    return cv2.addWeighted(initial_img, α, img, β, γ)

# Function to create a mask for white and yellow lane lines in HLS color space
def mask_white_yellow_hls(image):
    """
    Filters the image to isolate white and yellow colors for lane detection.
    
    Args:
    image (numpy.ndarray): Input RGB image.

    Returns:
    numpy.ndarray: Image with only white and yellow colors.
    """
    converted = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    # Define color range for white and yellow in HLS space
    lower_white = np.uint8([0, 200, 0])
    upper_white = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(converted, lower_white, upper_white)
    lower_yellow = np.uint8([10, 0, 100])
    upper_yellow = np.uint8([40, 255, 255])
    yellow_mask = cv2.inRange(converted, lower_yellow, upper_yellow)
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(image, image, mask=mask)

# Function to average detected lane line segments
def average_lanes(lines):
    """
    Calculates the average position and slope of detected left and right lane lines.
    
    Args:
    lines (list): List of line segments represented by (x1, y1, x2, y2).

    Returns:
    tuple: Averaged left and right lanes in the form of (slope, intercept).
    """
    left_lines, left_length, right_lines, right_length = [], [], [], []
    for line in lines:
        for col1, row1, col2, row2 in line:
            if col2 == col1:
                continue  # Skip vertical lines
            slope = (row2 - row1) / (col2 - col1)
            intercept = row1 - slope * col1
            length = np.sqrt((row2 - row1) ** 2 + (col2 - col1) ** 2)
            if slope < 0:
                left_lines.append((slope, intercept))
                left_length.append(length)
            else:
                right_lines.append((slope, intercept))
                right_length.append(length)
    left_lane, right_lane = None, None
    if len(left_length) > 0:
        left_lane = np.dot(left_length, left_lines) / np.sum(left_length)
    if len(right_length) > 0:
        right_lane = np.dot(right_length, right_lines) / np.sum(right_length)
    return left_lane, right_lane

# Function to get the coordinates of lane lines based on slope and intercept
def get_points(row_bottom, row_top, line):
    """
    Calculates the x-coordinates for given y-coordinates using the line's slope and intercept.
    
    Args:
    row_bottom (int): Y-coordinate for the bottom of the lane.
    row_top (int): Y-coordinate for the top of the lane.
    line (tuple): Slope and intercept of the line.

    Returns:
    tuple: Bottom and top points of the lane line.
    """
    if line is None:
        return None
    slope, intercept = line
    col_bottom = int((row_bottom - intercept) / slope)
    col_top = int((row_top - intercept) / slope)
    return ((col_bottom, int(row_bottom)), (col_top, int(row_top)))

# Function to detect and return left and right lane lines
def get_lane_lines(imshape, lines, row_top):
    """
    Extracts the left and right lane lines based on averaged line segments.
    
    Args:
    imshape (tuple): Shape of the image (height, width).
    lines (list): Detected lines from Hough Transformation.
    row_top (int): Top y-coordinate for lane lines.

    Returns:
    tuple: Coordinates of left and right lane lines.
    """
    left_lane, right_lane = average_lanes(lines)
    left_line = get_points(imshape[0], row_top, left_lane)
    right_line = get_points(imshape[0], row_top, right_lane)
    return left_line, right_line

# Function to overlay detected lane lines on the original image
def get_lane_lines_image(image, lines, color=[255, 0, 0], thickness=20):
    """
    Draws the detected lane lines onto an image.
    
    Args:
    image (numpy.ndarray): Original image.
    lines (tuple): Left and right lane lines.
    color (list): RGB color of the lane lines.
    thickness (int): Thickness of lane lines.

    Returns:
    numpy.ndarray: Image with lane lines drawn.
    """
    line_image = np.zeros_like(image)
    for line in lines:
        if line is not None:
            cv2.line(line_image, line[0], line[1], color, thickness)
    return cv2.addWeighted(image, 1.0, line_image, 0.95, 0.0)

# Variables to average lines over frames
QUEUE_MAX_LEN = 50
left_lines = collections.deque(maxlen=QUEUE_MAX_LEN)
right_lines = collections.deque(maxlen=QUEUE_MAX_LEN)

# Function to calculate average line positions across frames
def get_average_line(line, lines):
    """
    Averages line positions over multiple frames for smoother lane line display.
    
    Args:
    line (tuple): New line coordinates.
    lines (collections.deque): Deque of previous line positions.

    Returns:
    tuple: Averaged line coordinates.
    """
    if line is not None:
        lines.append(line)
    if len(lines) > 0:
        line = np.mean(lines, axis=0, dtype=np.int32)
        line = tuple(map(tuple, line))
    return line

# Main function to process each frame for lane detection
def process_image(image):
    """
    Processes an image to detect lane lines.
    
    Args:
    image (numpy.ndarray): Input RGB image.

    Returns:
    numpy.ndarray: Image with detected lane lines overlayed.
    """
    row_top = 330
    bottom_col_left, top_col_left, top_col_right = 100, 450, 560
    bottom_left = [bottom_col_left, image.shape[0]]
    top_left = [top_col_left, row_top]
    bottom_right = [image.shape[1], image.shape[0]]
    top_right = [top_col_right, row_top]
    hls_masked_image = mask_white_yellow_hls(image)
    gray = grayscale(hls_masked_image)
    blur_gray = gaussian_blur(gray, 15)
    edges = canny(blur_gray, 50, 150)
    vertices = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)
    h_lines = hough_lines(masked_edges, 1, np.pi / 180, 1, 5, 1)
    lane_lines = get_lane_lines(image.shape, h_lines, row_top)
    left_line = get_average_line(lane_lines[0], left_lines)
    right_line = get_average_line(lane_lines[1], right_lines)
    return get_lane_lines_image(image, (left_line, right_line))

# Process videos
white_output = os.path.join(output_dir, 'solidWhiteRight.mp4')
clip1 = VideoFileClip("test_videos/solidWhiteRight.mp4")
white_clip = clip1.fl_image(process_image)
white_clip.write_videofile(white_output, audio=False)

yellow_output = os.path.join(output_dir, 'solidYellowLeft.mp4')
clip2 = VideoFileClip("test_videos/solidYellowLeft.mp4")
yellow_clip = clip2.fl_image(process_image)
yellow_clip.write_videofile(yellow_output, audio=False)
