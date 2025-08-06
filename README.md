# ✨ Custom Canny Edge Detection Web App

This project is an interactive web application that detects edges in images using a **custom, from-scratch implementation** of the Canny edge detection algorithm.

It allows users to upload an image and instantly see the detected edges, showcasing a deep understanding of the underlying computer vision principles **without relying on pre-built library functions** for the core logic.

---

## 🚀 Features

- **Upload Images**: Users can upload images in various formats (JPG, PNG, etc.).
- **Real-Time Edge Detection**: Applies the custom Canny algorithm and displays the result instantly.
- **Side-by-Side View**: Shows the original and edge-detected images next to each other for easy comparison.
- **From-Scratch Algorithm**: The core edge detection logic is built using NumPy, demonstrating the fundamental steps of the Canny algorithm.

---

## 🛠️ Technologies Used

- **Python**: The core programming language.
- **Streamlit**: For creating and running the interactive web application.
- **NumPy**: For all numerical operations and implementing the Canny algorithm from scratch.
- **Pillow & OpenCV**: For image manipulation and handling.

---

## ⚙️ How to Run Locally

To run this application on your local machine, follow these steps:

1. **Clone the repository** (optional):
   ```bash
   git clone https://github.com/shriguruu/edge-detection.git
   cd YOUR_REPO_NAME
   ```

2. **Install the required libraries**:

    Make sure you have Python installed. Then, open your terminal and run:
   ```bash
   pip install streamlit numpy opencv-python-headless Pillow
   ```

3. **Run the Streamlit app**:

    In your terminal, navigate to the project directory and run:
   ```bash
   streamlit run app.py
   ```
   Your web browser will automatically open with the application running!




   
