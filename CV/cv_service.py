# This file will contain main logic for cv operations

from ultralytics import YOLO
from utils import read_video, create_video_writer
from trackers import Tracker
import cv2
import numpy as np
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
#from camera_movement_estimator import CameraMovementEstimator
#from view_transformer import ViewTransformer
#from speed_and_distance_estimator import SpeedAndDistance_Estimator
    
def main():

    video_frames_gen, fps = read_video('input_videos/bball_4.mp4')

    # Initialize services
    tracker = Tracker('models/models-new/best.pt')
    #team_assigner = TeamAssigner()
    #player_assigner = PlayerBallAssigner()

    tracks = []
    team_ball_control = []
    # May add frame count

    for frame_num, frame in enumerate(video_frames_gen):
        print(f'Processing frame {frame_num}')

        if frame_num == 0:
            # Initialize video writer
            frame_size = (frame.shape[1], frame.shape[0])  # (width, height)
            video_writer = create_video_writer('output_videos/bball_4_output2.avi', fps, frame_size)

            #team_assigner.assign_team_color(frame, tracks['players'][0])

        # Get tracks *Edit to recieve individual frames (Check video to see if prior frames/batches are necessary)
        tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=False,
                                       stub_path='stubs/track_stubs.pkl')
        
        # Get object positions 
        tracker.add_position_to_tracks(tracks)
        
        # Assign player teams

        # Assign ball acquisition

        # Draw annotations on the current frame *Edit to do individual frames (may be functional already)
        annotated_frame = tracker.draw_annotations(frame, tracks, frame_num, team_ball_control)

        # Write the annotated frame directly to the video
        video_writer.write(annotated_frame)

    # Release the video writer after processing all frames
    video_writer.release()

if __name__ == '__main__':
    main()