import cv2
import os


def save_frames(video_path, output_dir, full_name, target_res=(1920, 1080), num_frames=20):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    name = full_name.split('.')[0]

    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate the frame indices to extract
    indices = [int(i * frame_count / (num_frames - 1)) for i in range(num_frames)]

    # Iterate through the selected frame indices
    for index in indices:
        # Set the frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)

        # Read the frame
        ret, frame = cap.read()

        if ret:
            # Resize the frame to the target resolution
            frame = cv2.resize(frame, target_res)

            # Create the output file name
            output_file = os.path.join(output_dir, f"{name}_frame_{index}.jpg")

            # Save the frame as a JPG image
            cv2.imwrite(output_file, frame)

    # Release the video capture object
    cap.release()


if __name__ == "__main__":
    # Set the correct input video file path, output directory, and target resolution
    input_folder = "extract/vids"
    output_directory = "extract/frames"

    videos = {"Berghouse Leopard Jog.mp4", "DJI_0596.MP4", "DJI_0790.MOV", "DJI_0862.MOV", "Stockflue Flyaround.mp4"}

    target_resolution = (1920, 1080)
    num_frames = 25

    for video in videos:
        video_path = f"{input_folder}/{video}"

        # Save frames from the video
        save_frames(video_path, output_directory, video, target_resolution, num_frames)
