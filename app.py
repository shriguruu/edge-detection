import streamlit as st
import numpy as np
import cv2 # Used for resizing the image
from PIL import Image
from math import sqrt, atan2, degrees, pi

# --- Page Configuration ---
st.set_page_config(
    page_title="Custom Canny Edge Detection",
    page_icon="✨",
    layout="wide",
)

# --- Custom Canny Edge Detection Implementation ---

def rgb2gray(img):
    """Converts an RGB image to grayscale."""
    return np.dot(img[..., :3], [0.299, 0.587, 0.114])

def gaussian_kernel(size, sigma=1):
    """Creates a Gaussian kernel."""
    ax = np.linspace(-(size // 2), size // 2, size)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)

def convolve(img, kernel):
    """Performs convolution between an image and a kernel."""
    kh, kw = kernel.shape
    ih, iw = img.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant')
    output = np.zeros_like(img)

    for i in range(ih):
        for j in range(iw):
            region = padded[i:i+kh, j:j+kw]
            output[i, j] = np.sum(region * kernel)
    return output

def sobel_filters(img):
    """Applies Sobel filters to find gradients."""
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    Gx = convolve(img, Kx)
    Gy = convolve(img, Ky)

    magnitude = np.hypot(Gx, Gy)
    magnitude = magnitude / magnitude.max() * 255
    direction = np.arctan2(Gy, Gx)
    return magnitude, direction

def non_max_suppression(mag, angle):
    """Thins edges by performing non-maximum suppression."""
    H, W = mag.shape
    output = np.zeros((H, W), dtype=np.int32)
    angle = angle * 180.0 / np.pi
    angle[angle < 0] += 180

    for i in range(1, H-1):
        for j in range(1, W-1):
            q = 255
            r = 255
            # Angle 0
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q = mag[i, j+1]
                r = mag[i, j-1]
            # Angle 45
            elif 22.5 <= angle[i,j] < 67.5:
                q = mag[i+1, j-1]
                r = mag[i-1, j+1]
            # Angle 90
            elif 67.5 <= angle[i,j] < 112.5:
                q = mag[i+1, j]
                r = mag[i-1, j]
            # Angle 135
            elif 112.5 <= angle[i,j] < 157.5:
                q = mag[i-1, j-1]
                r = mag[i+1, j+1]

            if mag[i,j] >= q and mag[i,j] >= r:
                output[i,j] = mag[i,j]
            else:
                output[i,j] = 0
    return output

def threshold(img, lowThresholdRatio=0.1, highThresholdRatio=0.25):
    """Applies double thresholding to identify strong, weak, and non-edges."""
    high = img.max() * highThresholdRatio
    low = high * lowThresholdRatio
    strong = 255
    weak = 25
    res = np.zeros_like(img)
    strong_i, strong_j = np.where(img >= high)
    weak_i, weak_j = np.where((img <= high) & (img >= low))
    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak
    return res, weak, strong

def hysteresis(img, weak, strong=255):
    """Transforms weak pixels into strong ones if they are connected to strong pixels."""
    H, W = img.shape
    for i in range(1, H-1):
        for j in range(1, W-1):
            if img[i, j] == weak:
                if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or
                    (img[i+1, j+1] == strong) or (img[i, j-1] == strong) or
                    (img[i, j+1] == strong) or (img[i-1, j-1] == strong) or
                    (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                    img[i, j] = strong
                else:
                    img[i, j] = 0
    return img

def canny_edge_detector(image):
    """The main function that orchestrates the Canny edge detection process."""
    gray = rgb2gray(image)
    blur = convolve(gray, gaussian_kernel(5, 1))
    mag, angle = sobel_filters(blur)
    nms = non_max_suppression(mag, angle)
    thresh, weak, strong = threshold(nms)
    result = hysteresis(thresh, weak, strong)
    return result

# --- Streamlit UI ---

st.title("✨ Custom Canny Edge Detection")
st.write(
    "This application uses a from-scratch implementation of the Canny algorithm to detect edges in an image. "
    "Upload an image to see it work!"
)
st.markdown("---")

uploaded_file = st.file_uploader(
    "Choose an image file", type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        original_image = np.array(image)

        st.markdown("### Original Image vs. Custom Canny Edges")
        col1, col2 = st.columns(2)

        with col1:
            st.image(original_image, caption="Original Image", use_container_width=True)

        # Resize for faster processing, especially for large images
        resized_img = cv2.resize(original_image, (256, 256))
        
        # Apply the custom Canny edge detector
        edges = canny_edge_detector(resized_img)

        with col2:
            st.image(edges, caption="Custom Canny Edges", use_container_width=True)

        st.success("Edge detection complete using the custom algorithm!")

    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.info("Upload an image to get started.")

st.markdown("---")

