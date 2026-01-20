import cv2


# Create a function to map landmark points
def landmark_points(det, PADDING, landmark):
    xmin = det.xmin
    xmax = det.xmax
    ymin = det.ymin
    ymax = det.ymax
    padding = PADDING
    slope_x = (xmax + padding) - (xmin - padding)
    slope_y = (ymax + padding) - (ymin - padding)
    xnorm = xmin - padding + slope_x * landmark.x
    ynorm = ymin - padding + slope_y * landmark.y
    return xnorm, ynorm


# Create a function to map bounding box coordinates
def bbox_keypoints(det, w, h):
    x1 = int(det.xmin * w)
    y1 = int(det.ymin * h)
    x2 = int(det.xmax * w)
    y2 = int(det.ymax * h)
    return x1, y1, x2, y2


# Create a function to extract keypoints for pose classification
def extract_keypoints(landmarks, detections, body_idx, PADDING):
    keypoints = []
    det = None
    if detections is not None and body_idx < len(detections.detections):
        det = detections.detections[body_idx]

    for i, landmark in enumerate(landmarks):
        confidence = getattr(landmark, 'confidence', 0.6)
        if det is not None:
            xnorm, ynorm = landmark_points(det, PADDING, landmark)
        else:
            xnorm = landmark.x
            ynorm = landmark.y

        confidence = getattr(landmark, 'confidence', 0.6)

        keypoints.append((xnorm, ynorm, confidence))
    return keypoints


# Create a function to display
def draw_pose(frame, pose, x1, y1):
    if pose is None:
        return None

    label = f" Pose: {pose.pose_name}, Confidence: {pose.confidence:.2f}"

    cv2.putText(frame, label, (x1, y1 + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)


# TODO: Create a function to move the robot according to the detected pose
#       Must be implemented below draw_pose() within the if statement


# Create a funcion for detecting a human, their pose and display their skeleton
def human_pose_recognition(frame, detections, bodyPose, pose_classifier,
                           PADDING, skeleton_edges, CONFIDENCE_THRESHOLD):
    # If there are no frames, return None
    if frame is None:
        return None

    # Copy the frame and extract its height and width
    display_frame = frame.copy()
    h, w = display_frame.shape[:2]

    # If body landmarks exist, draw a body skeleton
    if bodyPose is not None:
        try:
            # Determine if keypoint and prediction messages exist
            if hasattr(bodyPose, 'gathered') and bodyPose.gathered:
                for body_idx, msg_group in enumerate(bodyPose.gathered):
                    if hasattr(msg_group, 'keypoints') and msg_group.keypoints:
                        landmarks = msg_group.keypoints
                        points = []

                        # Define variables used in body skeleton mapping
                        dets = None
                        bbox_coords = None
                        if detections is not None and body_idx < len(detections.detections):
                            dets = detections.detections[body_idx]
                            bbox_coords = bbox_keypoints(dets, w, h)

                        # Classification keypoints
                        classificationKeypoints = extract_keypoints(landmarks, detections, body_idx, PADDING)
                        classifiedPose = pose_classifier.classifyPose(classificationKeypoints)

                        print(f"pose: {classifiedPose}")

                        # Draw body landmark points
                        for i, landmark in enumerate(landmarks):
                            if dets is not None:
                                xnorm, ynorm = landmark_points(dets, PADDING, landmark)
                                x = int(xnorm * w)
                                y = int(ynorm * h)
                            else:
                                x = int(landmark.x * w)
                                y = int(landmark.y * h)

                            # Display the connection points of the skeleton
                            cv2.circle(display_frame, (x, y), 5, (0, 0, 255), -1)
                            points.append((x, y))

                        # Draw body skeleton connections
                        if len(points) > 0 and skeleton_edges:
                            for edge in skeleton_edges:
                                start_idx, end_idx = edge
                                if start_idx < len(points) and end_idx < len(points):
                                    cv2.line(display_frame, points[start_idx], points[end_idx], (255, 255, 255), 2)

                        if classifiedPose and bbox_coords:
                            draw_pose(display_frame,
                                      classifiedPose,
                                      bbox_coords[0],
                                      bbox_coords[1],)

        except Exception as e:
            print(f"Cannot draw landmarks: {e}")
    if detections is not None and len(detections.detections) > 0:
        for det in detections.detections:
            if det.confidence > CONFIDENCE_THRESHOLD:
                try:
                    # Determine the key points of the bounding box and scale accordingly
                    x1, y1, x2, y2 = bbox_keypoints(det, w, h)

                    # Visualise the bounding box
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                    # Determine the human detection confidence score
                    confidence_score = f"Human: {det.confidence * 100:.0f}%"

                    # Determine spatial coordinates
                    if hasattr(det, 'spatialCoordinates'):
                        x = det.spatialCoordinates.x / 1000
                        y = det.spatialCoordinates.y / 1000
                        z = det.spatialCoordinates.z / 1000
                        x_display = f"x: {x:.2f}m"
                        y_display = f"y: {y:.2f}m"
                        z_display = f"z: {z:.2f}m"

                    # Visualise confidence score and spatial coordinates
                    cv2.putText(display_frame, confidence_score, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(display_frame, x_display, (x1, y1 + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    cv2.putText(display_frame, y_display, (x1, y1 + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    cv2.putText(display_frame, z_display, (x1, y1 + 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                    # print(f"Distance of the human from the camera is: {z_display}.")
                except Exception as e:
                    print(f"Error drawing bounding box: {e}")
    return display_frame
