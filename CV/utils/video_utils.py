import cv2

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    
    # Get frame rate of input video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Create a generator to yield frames
    def frame_generator():
        a = 0
        while True:
            a += 1
            ret, frame = cap.read()

            # Break loop if video done
            if not ret:
                break

            yield frame

    return frame_generator(), fps

# Create a video writer to svae frames one at a time
def create_video_writer(output_video_path, fps, frame_size):
    # Set video format
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # Initialize video writer with output path, codec, fps, and frame size
    out = cv2.VideoWriter(output_video_path, fourcc, fps, frame_size)

    return out