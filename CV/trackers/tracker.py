from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import sys 
sys.path.append('../') # used to import from utils
from utils import get_center_of_bbox, get_bbox_width, get_foot_position

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path) 
        self.tracker = sv.ByteTrack()
        self.tracks={
            "players":[],
            "ball":[]
        }

    def add_position_to_tracks(self,tracks,frame_num):
        for object, object_tracks in tracks.items():
            track = object_tracks[frame_num]
            for track_id, track_info in track.items():
                bbox = track_info['bbox']
                if object == 'ball':
                    position = get_center_of_bbox(bbox)
                else:
                    position = get_foot_position(bbox)
                tracks[object][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self,ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Interpolate missing values
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    # Changed to recieve one frame
    def detect_frame(self, frame):
        return self.model.predict(frame, conf=0.1)

    def get_object_tracks(self, frame, frame_num, read_from_stub=False, stub_path=None):
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detection = self.detect_frame(frame)[0]
            
        cls_names = detection.names
        cls_names_inv = {v:k for k,v in cls_names.items()}

        # Covert to supervision Detection format
        detection_supervision = sv.Detections.from_ultralytics(detection)

        # Track Objects
        detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

        self.tracks["players"].append({})
        self.tracks["ball"].append({})

        # Track Player
        for frame_detection in detection_with_tracks:
            bbox = frame_detection[0].tolist()
            cls_id = frame_detection[3]
            track_id = frame_detection[4]

            if cls_id == cls_names_inv['Player']:
                self.tracks["players"][frame_num][track_id] = {"bbox":bbox}

        # Track Ball    
        for frame_detection in detection_supervision:
            bbox = frame_detection[0].tolist()
            cls_id = frame_detection[3]

            if cls_id == cls_names_inv['Ball']:
                self.tracks["ball"][frame_num][1] = {"bbox":bbox}

        # Save stub * This needs to be moved to main so that it is called on finish, Or changed to append frames to the stub? (This would require a freah stub for each run)
        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(self.tracks,f)

        return self.tracks
    
    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame

    def draw_traingle(self,frame,bbox,color):
        y= int(bbox[1])
        x,_ = get_center_of_bbox(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame

    def draw_team_ball_control(self,frame,frame_num,team_ball_control):
        # Draw a semi-transparent rectaggle 
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900,970), (255,255,255), -1 )
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        team_ball_control_till_frame = team_ball_control[:frame_num+1]
        # Get the number of time each team had ball control
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]

        if team_1_num_frames + team_2_num_frames != 0:
            team_1 = team_1_num_frames/(team_1_num_frames+team_2_num_frames)
            team_2 = team_2_num_frames/(team_1_num_frames+team_2_num_frames)
        else:
            team_1 = 0
            team_2 = 0

        #cv2.putText(frame, f"Team 1 Ball Control: {team_1*100:.2f}%",(1400,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        #cv2.putText(frame, f"Team 2 Ball Control: {team_2*100:.2f}%",(1400,950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        cv2.putText(frame, f"Ball Control: {team_ball_control[frame_num]}",(1400,900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)

        return frame

    def draw_annotations(self, input_frame, track, frame_num, team_ball_control):
        frame = input_frame.copy()

        player_dict = track["players"][frame_num]
        ball_dict = track["ball"][frame_num]

        # Draw Players
        for track_id, player in player_dict.items():
            color = player.get("team_color",(255,0,0))
            frame = self.draw_ellipse(frame, player["bbox"],color, track_id)

            if player.get('has_ball',False):
                frame = self.draw_traingle(frame, player["bbox"],(0,0,255))
                
        # Draw ball 
        for track_id, ball in ball_dict.items():
            frame = self.draw_traingle(frame, ball["bbox"],(0,255,0))


        # Draw Team Ball Control
        #frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

        return frame
