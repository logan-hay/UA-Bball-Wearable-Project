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

    video_frames_gen, fps = read_video('input_videos/duke.mp4')

    # Initialize services
    tracker = Tracker('models/models-new/best.pt')
    team_assigner = TeamAssigner()
    #player_assigner = PlayerBallAssigner()

    tracks = []
    team_ball_control = []

    for frame_num, frame in enumerate(video_frames_gen):
        print(f'Processing frame {frame_num}')

        # Track objects
        tracks = tracker.get_object_tracks(frame,
                                        frame_num,
                                        read_from_stub=False,
                                        stub_path='stubs/track_stubs.pkl')
        
        if frame_num == 0: # These both only need to be done once
            # Initialize video writer
            frame_size = (frame.shape[1], frame.shape[0])  # (width, height)
            video_writer = create_video_writer('output_videos/fbf_duke.avi', fps, frame_size)

            # Assign player teams
            team_assigner.assign_team_color(frame, 
                                        tracks['players'][0])
        
        # Get object positions | Not sure what this was used for, Maybe debugging?
        #tracker.add_position_to_tracks(tracks)
    
        for player_id, track in tracks['players'][frame_num].items():
            team = team_assigner.get_player_team(frame,   
                                                track['bbox'],
                                                player_id)
            tracks['players'][frame_num][player_id]['team'] = team 
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]
            color = tracks['players'][frame_num][player_id]['team_color']

        # Assign ball acquisition
            
        # Draw annotations on the current frame
        annotated_frame = tracker.draw_annotations(frame, tracks, frame_num, team_ball_control)

        # Write the annotated frame directly to the video
        video_writer.write(annotated_frame)

    # Release the video writer after processing all frames
    video_writer.release()

if __name__ == '__main__':
    main()