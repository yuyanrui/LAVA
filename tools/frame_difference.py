import cv2
import numpy as np

def preprocess_frame(frame, kernel_size=(5, 5), sigma=0, width=640, height=360):
    # Convert frame to grayscale
    frame = cv2.resize(frame, (width,height) )
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to the grayscale frame
    blurred_frame = cv2.GaussianBlur(gray_frame, kernel_size, sigma)

    return blurred_frame

def calculate_frame_difference(frame1, frame2,thresh=48):
    # Calculate the absolute difference between the two preprocessed frames
    difference = cv2.absdiff(frame1, frame2)

    # Threshold the difference to create a binary image (only black and white pixels)
    _, thresholded_difference = cv2.threshold(difference,thresh, 255, cv2.THRESH_BINARY)

    return thresholded_difference

def frame_difference_score(frame1, frame2, thresh=48,width=640,height=360):

    frame1 = preprocess_frame(frame1, width=width,height=height)
    frame2 = preprocess_frame(frame2, width=width,height=height)
    thresholded_difference = calculate_frame_difference(frame1,frame2,thresh=thresh)

    return np.sum(thresholded_difference)/((width*height)*255)
    

def main():
    # Load the video
    video_path = \
        "XXX.mp4"
    cap = cv2.VideoCapture(video_path)

    # Read the first frame and preprocess it
    ret, previous_frame = cap.read()
    previous_frame = preprocess_frame(previous_frame)

    while True:
        # Read the next frame and preprocess it
        cap.set(cv2.CAP_PROP_POS_FRAMES, 10000)
        ret, current_frame = cap.read()

        # Break the loop if we have reached the end of the video
        if not ret:
            break

        current_frame = preprocess_frame(current_frame)

        # Calculate the difference between two consecutive frames
        frame_difference = calculate_frame_difference(previous_frame, current_frame)

        # Show the original frame and the difference frame
        cv2.imwrite("Original.jpg", current_frame)
        cv2.imwrite("Difference.jpg", frame_difference)

        previous_frame = current_frame
        exit()
    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
