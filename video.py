import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import cv2
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
import math
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8x.pt')  # Ensure this path is correct

# Initialize DeepSORT
deepsort = DeepSort()

# Function to calculate the Euclidean distance between two points
def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# Function to detect players using YOLOv8
def detect_players(frame):
    # Perform inference
    results = model(frame)
    
    # Extract results
    boxes = results[0].boxes.xyxy.cpu().numpy()  # Get the bounding boxes
    confidences = results[0].boxes.conf.cpu().numpy()  # Get the confidence scores
    
    # Prepare data for DeepSORT
    bboxes = []
    for i, bbox in enumerate(boxes):
        x1, y1, x2, y2 = bbox
        conf = confidences[i]  # Get the confidence score for each bounding box
        bboxes.append([x1, y1, x2, y2])  # Use list format [x1, y1, x2, y2]
    
    return bboxes, confidences

# Function to track players and calculate distances
def track_and_calculate_distance(video_path):
    # Initialize video capture
    video = cv2.VideoCapture(video_path)

    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        
        # Detect players in the current frame
        bboxes, confidences = detect_players(frame)
        
        # Convert detections to DeepSORT format
        detections = [(bbox, conf) for bbox, conf in zip(bboxes, confidences)]
        
        # Convert detections to DeepSORT-compatible format
        raw_detections = [[bbox, conf] for bbox, conf in detections]
        
        # Update tracks with the raw detection data
        tracks = deepsort.update_tracks(raw_detections, frame=frame)
        
        # Ensure that at least two players are tracked
        visible_tracks = [track for track in tracks if track.is_confirmed() and track.time_since_update == 0]
        
        if len(visible_tracks) >= 2:
            # Use first two visible tracks
            track1, track2 = visible_tracks[0], visible_tracks[1]
            
            # Get the bounding boxes and centers of the first two players
            person1_bbox = track1.to_tlbr()
            person2_bbox = track2.to_tlbr()
            
            person1_center = (int((person1_bbox[0] + person1_bbox[2]) / 2), 
                              int((person1_bbox[1] + person1_bbox[3]) / 2))
            person2_center = (int((person2_bbox[0] + person2_bbox[2]) / 2), 
                              int((person2_bbox[1] + person2_bbox[3]) / 2))
            
            # Get track IDs
            person1_id = track1.track_id
            person2_id = track2.track_id
            
            # Calculate the Euclidean distance between the two centers
            distance = calculate_distance(person1_center, person2_center)
            
            # Draw bounding boxes for both players
            cv2.rectangle(frame, (int(person1_bbox[0]), int(person1_bbox[1])), (int(person1_bbox[2]), int(person1_bbox[3])), (0, 255, 0), 2)
            cv2.rectangle(frame, (int(person2_bbox[0]), int(person2_bbox[1])), (int(person2_bbox[2]), int(person2_bbox[3])), (0, 255, 0), 2)
            
            # Draw the centers of the players
            cv2.circle(frame, person1_center, 5, (255, 0, 0), -1)
            cv2.circle(frame, person2_center, 5, (255, 0, 0), -1)
            
            # Draw a line between the two players
            cv2.line(frame, person1_center, person2_center, (0, 0, 255), 2)
            
            # Display the distance and player tracking IDs on the frame
            cv2.putText(frame, f"ID: {person1_id}", (int(person1_bbox[0]), int(person1_bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"ID: {person2_id}", (int(person2_bbox[0]), int(person2_bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Distance: {int(distance)} px", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Display the frame with detected players and distance
        cv2.imshow("Tracking & Distance", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

# Load the video
video_path = 'C:/Users/vvv40/OneDrive/Desktop/video tracking/video.mp4'

# Track players and calculate the distance between them
track_and_calculate_distance(video_path)
