import cv2
import os

def extract_frames(video_path, output_folder):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames.")

if __name__ == "__main__":
    video_path = "example.mp4"
    output_folder = "frames"
    print(video_path)
    os.makedirs(output_folder, exist_ok=True)
    extract_frames(video_path, output_folder)
