import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class poseDetection:
    """Data stored from the result of the pose detection."""
    pose_name: str
    confidence: float
    details: Dict[str, float]


class poseKeypoints:
    def __init__(self, confidence_threshold: float = 0.3):
        """Initialise the pose classifier.

        Args:
            confidence_threshold: Minimum confidence required for detection
        """
        # Define confidence threshold
        self.confidence_threshold = confidence_threshold

        # Coco dataset keypoints
        self.Keypoints = {
            'nose': 0, 'left-eye': 1, 'right-eye': 2,
            'left-ear': 3, 'right-ear': 4, 'left-shoulder': 5,
            'right-shoulder': 6, 'left-elbow': 7, 'right-elbow': 8,
            'left-wrist': 9, 'right-wrist': 10, 'left-hip': 11,
            'right-hip': 12, 'left-knee': 13, 'right-knee': 14,
            'left-ankle': 15, 'right-ankle': 16
        }

    def calculateAngle(self, a, b, c):
        """Calculate the angle of the vertex point (b) using three points.

        Args:
            a, b, c: Points as (x, y) tuples

        Returns:
            Angle in degrees
        """
        # Convert the tuple arguments to numpy arrays
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        # Calculate the differences between points
        ba = a - b
        bc = c - b

        # Calculate the angle (in degrees) of the vertex point (b)
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba)
                                         * np.linalg.norm(bc))
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)

    def calculateHorizontalAngle(self, a, b):
        """Calculates the angle to the horizontal using two points.

        Args:
            a, b: Points as (x, y) tuples

        Returns:
                Angle in degrees
        """
        # Convert the tuple arguments to numpy arrays
        a = np.array(a)
        b = np.array(b)

        # Calculate differences between points
        # ab[0] = horizontal difference, ab[1] = vertical difference
        ab = b - a

        # Return the angle (in degrees) to the horizontal
        angle = abs(np.degrees(np.arctan2(ab[1], ab[0])))
        return min(angle, 180 - angle)

    def parse_keypoints(self, keypoints: List[Tuple[float, float, float]]) -> Dict[int, Tuple[float, float]]:
        """Parse keypoints and filter by confidence.

        Args:
            keypoints: List of (x, y, confidence) tuples

        Returns:
            Dictionary mapping keypoint index to (x, y) position
        """
        points = {}

        # If the keypoint confidence is greater than the confidence threshold,
        # store it as a point
        for idx, (x, y, conf) in enumerate(keypoints):
            if conf > self.confidence_threshold:
                points[idx] = (x, y)
        return points

    def tpose_gesture(self, points: Dict[int, Tuple[float, float]]) -> Optional[poseDetection]:
        """Detect if a T-Pose body pose has been performed.

        Args:
            points: Dictionary mapping keypoint index to (x, y) position

        Returns:
            poseDetection object if detected, otherwise None
        """
        # Extract required keypoints from the Coco dataset keypoints
        K = self.Keypoints
        keypoints = [K['left-shoulder'], K['right-shoulder'],
                     K['left-elbow'], K['right-elbow'],
                     K['left-wrist'], K['right-wrist']]

        # If not all of the keypoints are present, return None
        if not all(idx in points for idx in keypoints):
            return None

        # Determine the angle of the arm created by the elbow
        left_elbow_angle = self.calculateAngle(points[keypoints[0]],
                                               points[keypoints[2]],
                                               points[keypoints[4]])
        right_elbow_angle = self.calculateAngle(points[keypoints[1]],
                                                points[keypoints[3]],
                                                points[keypoints[5]])

        # Determine the horizontal angle of the arm
        left_arm_angle = self.calculateHorizontalAngle(points[keypoints[0]],
                                                       points[keypoints[4]])
        right_arm_angle = self.calculateHorizontalAngle(points[keypoints[1]],
                                                        points[keypoints[5]])

        # Determine if the arm is straight [130,180]°
        left_arm_straight = 130 <= left_elbow_angle <= 180
        right_arm_straight = 130 <= right_elbow_angle <= 180

        # Determine if the arm is horizontal [0,45]°
        left_arm_horizontal = left_arm_angle <= 45
        right_arm_horizontal = right_arm_angle <= 45

        # If both arms are straight and horizontal, determine the confidences
        # and store the data in the data class, otherwise return None
        if left_arm_straight and right_arm_straight and left_arm_horizontal and right_arm_horizontal:
            angle_confidence = (left_elbow_angle + right_elbow_angle) / 360
            horizontal_confidence = 1.0 - ((left_arm_angle + right_arm_angle) / 90)
            confidence = (angle_confidence + horizontal_confidence) / 2.0
            return poseDetection(
                pose_name='T-Pose',
                confidence=min(confidence, 1.0),
                details={
                    'left_elbow_angle': left_elbow_angle,
                    'right_elbow_angle': right_elbow_angle,
                    'left_arm_angle': left_arm_angle,
                    'right_arm_angle': right_arm_angle,
                }
            )
        else:
            return None

    def handsup_gesture(self, points: Dict[int, Tuple[float, float]]) -> Optional[poseDetection]:
        """Detect if a hands up body pose has been performed.

        Args:
            points: Dictionary mapping keypoint index to (x, y) position

        Returns:
            poseDetection object if detected, otherwise None
        """
        # Extract required keypoints from the Coco dataset keypoints
        K = self.Keypoints
        keypoints = [K['left-shoulder'], K['right-shoulder'],
                     K['left-elbow'], K['right-elbow'],
                     K['left-wrist'], K['right-wrist']]

        # If not all of the required keypoints are present, return None
        if not all(idx in points for idx in keypoints):
            return None

        # Determine if the wrist is above the shoulder
        left_wrist_above = points[keypoints[4]][1] < points[keypoints[0]][1]
        right_wrist_above = points[keypoints[5]][1] < points[keypoints[1]][1]
        if not (left_wrist_above and right_wrist_above):
            return None

        # Determine the height of the wrist
        left_wrist_height = points[keypoints[0]][1] - points[keypoints[4]][1]
        right_wrist_height = points[keypoints[1]][1] - points[keypoints[5]][1]

        # Determine the orientation of the wrist w.r.t. the shoulder
        left_wrist_orient = abs(points[keypoints[4]][0] -
                                points[keypoints[0]][0])
        right_wrist_orient = abs(points[keypoints[5]][0] -
                                 points[keypoints[1]][0])

        # Determine the angle of the arm created by the elbow and its
        # confidence, otherwise return None
        angle_confidence = 1.0
        if keypoints[2] in points and keypoints[3] in points:
            left_elbow_angle = self.calculateAngle(points[keypoints[0]],
                                                   points[keypoints[2]],
                                                   points[keypoints[4]])
            right_elbow_angle = self.calculateAngle(points[keypoints[1]],
                                                    points[keypoints[3]],
                                                    points[keypoints[5]])
            angle_confidence = (left_elbow_angle + right_elbow_angle) / 360.0
        else:
            return None

        # If the wrist is above a specific height, determine the confidences
        # and store the data in the data class, otherwise return None
        if left_wrist_height > 0.2 and right_wrist_height > 0.2:
            height_confidence = min((left_wrist_height + right_wrist_height) / 1.0, 1.0)
            orientation_confidence = 1.0 - min((left_wrist_orient + right_wrist_orient) / 1.0, 1.0)
            confidence = (angle_confidence + height_confidence + orientation_confidence) / 3.0
            return poseDetection(
                pose_name='Both Arms Up',
                confidence=min(confidence, 1.0),
                details={
                    'left_wrist_height': left_wrist_height,
                    'right_wrist_height': right_wrist_height,
                    'left_wrist_orientation': left_wrist_orient,
                    'right_wrist_orientation': right_wrist_orient,
                }
            )
        else:
            return None

    def leftarmwide_gesture(self, points: Dict[int, Tuple[float, float]]) -> Optional[poseDetection]:
        """Detect if a wide left arm body pose has been performed.

        Args:
            points: Dictionary mapping keypoint index to (x, y) position

        Returns:
            poseDetection object if detected, otherwise None
        """
        # Extract required keypoints from Coco dataset keypoints
        K = self.Keypoints
        keypoints = [K['left-shoulder'], K['right-shoulder'],
                     K['left-elbow'], K['right-elbow'],
                     K['left-wrist'], K['right-wrist']]

        # If not all of the required keypoints are present, return None
        if not all(idx in points for idx in keypoints):
            return None

        # Determine the angle of the left arm created by the left elbow
        left_elbow_angle = self.calculateAngle(points[keypoints[0]],
                                               points[keypoints[2]],
                                               points[keypoints[4]])

        # Determine the horizontal angle of the left arm
        left_arm_angle = self.calculateHorizontalAngle(points[keypoints[0]],
                                                       points[keypoints[4]])

        # Determine if the left arm is straight [130,180]°
        left_arm_straight = 130 <= left_elbow_angle <= 180

        # Determine if the left arm is horizontal [0,45]°
        left_arm_horizontal = left_arm_angle <= 45

        # Ensure the left arm is straight and horizontal, otherwise return None
        if not (left_arm_straight and left_arm_horizontal):
            return None

        # Determine the angle of the right arm created by the right elbow
        right_elbow_angle = self.calculateAngle(points[keypoints[1]],
                                                points[keypoints[3]],
                                                points[keypoints[5]])

        # Determine the horizontal angle of the right arm
        right_arm_angle = self.calculateHorizontalAngle(points[keypoints[1]],
                                                        points[keypoints[5]])

        # Determine if the right arm is straight [120,180]°
        right_arm_straight = 120 <= right_elbow_angle <= 180

        # Determine if the right arm is horizontal [0,45]°
        right_arm_horizontal = right_arm_angle <= 45

        # If the right arm is both straight and horizontal, return None
        if right_arm_straight and right_arm_horizontal:
            return None

        # Determine if the right wrist is above the right shoulder and
        # if it is, return None
        right_wrist_above = points[keypoints[5]][1] < points[keypoints[1]][1]
        if right_wrist_above:
            return None

        # If the arm is straight and horizontal, determine the confidences
        # and store the data in the data class, otherwise return None
        if left_arm_straight and left_arm_horizontal:
            angle_confidence = left_elbow_angle / 180
            horizontal_confidence = 1.0 - (left_arm_angle / 45)
            confidence = (angle_confidence + horizontal_confidence) / 2.0
            return poseDetection(
                pose_name='Left Arm Wide',
                confidence=min(confidence, 1.0),
                details={
                    'left_elbow_angle': left_elbow_angle,
                    'left_arm_angle': left_arm_angle,
                }
            )
        else:
            return None

    def rightarmwide_gesture(self, points: Dict[int, Tuple[float, float]]) -> Optional[poseDetection]:
        """Detect if a wide right arm body pose has been performed.

        Args:
            points: Dictionary mapping keypoint index to (x, y) position

        Returns:
            poseDetection object if detected, otherwise None
        """
        # Extract required keypoints from Coco dataset keypoints
        K = self.Keypoints
        keypoints = [K['left-shoulder'], K['right-shoulder'],
                     K['left-elbow'], K['right-elbow'],
                     K['left-wrist'], K['right-wrist']]

        # If not all of the required keypoints are present, return None
        if not all(idx in points for idx in keypoints):
            return None

        # Determine the angle of the right arm created by the right elbow
        right_elbow_angle = self.calculateAngle(points[keypoints[1]],
                                                points[keypoints[3]],
                                                points[keypoints[5]])

        # Determine the horizontal angle of the right arm
        right_arm_angle = self.calculateHorizontalAngle(points[keypoints[1]],
                                                        points[keypoints[5]])

        # Determine if the right arm is straight [130,180]°
        right_arm_straight = 130 <= right_elbow_angle <= 180

        # Determine if the right arm is horizontal [0,45]°
        right_arm_horizontal = right_arm_angle <= 45

        # Determine the angle of the left arm created by the left elbow
        left_elbow_angle = self.calculateAngle(points[keypoints[0]],
                                               points[keypoints[2]],
                                               points[keypoints[4]])

        # Determine the horizontal angle of the left arm
        left_arm_angle = self.calculateHorizontalAngle(points[keypoints[0]],
                                                       points[keypoints[4]])

        # Determine if the left arm is straight [120,180]°
        left_arm_straight = 120 <= left_elbow_angle <= 180

        # Determine if the left arm is horizontal [0,45]°
        left_arm_horizontal = left_arm_angle <= 45

        # If the left arm is both straight and horizontal, return None
        if left_arm_straight and left_arm_horizontal:
            return None

        # Determine if the left wrist is above the left shoulder and
        # if it is, return None
        left_wrist_above = points[keypoints[4]][1] < points[keypoints[0]][1]
        if left_wrist_above:
            return None

        # If the arm is straight and horizontal, determine the confidences
        # and store the data in the data class, otherwise return None
        if right_arm_straight and right_arm_horizontal:
            angle_confidence = right_elbow_angle / 180
            horizontal_confidence = 1.0 - (right_arm_angle / 45)
            confidence = (angle_confidence + horizontal_confidence) / 2.0
            return poseDetection(
                pose_name='Right Arm Wide',
                confidence=min(confidence, 1.0),
                details={
                    'right_elbow_angle': right_elbow_angle,
                    'right_arm_angle': right_arm_angle,
                }
            )
        else:
            return None

    def leftarm_up(self, points: Dict[int, Tuple[float, float]]) -> Optional[poseDetection]:
        """Detect if a left hand up body pose has been performed.

        Args:
            points: Dictionary mapping keypoint index to (x, y) position

        Returns:
            poseDetection object if detected, otherwise None
        """
        # Extract required keypoints from Coco dataset keypoints
        K = self.Keypoints
        keypoints = [K['left-shoulder'], K['right-shoulder'],
                     K['left-elbow'], K['right-elbow'],
                     K['left-wrist'], K['right-wrist']]

        # If not all of the required keypoints are present, return None
        if not all(idx in points for idx in keypoints):
            return None

        # Determine if the left wrist is above the left shoulder
        left_wrist_above = points[keypoints[4]][1] < points[keypoints[0]][1]
        if not left_wrist_above:
            return None

        # Determine the height of the left wrist
        left_wrist_height = points[keypoints[0]][1] - points[keypoints[4]][1]

        # Determine the orientation of the left wrist w.r.t. the left shoulder
        left_wrist_orient = abs(points[keypoints[4]][0] -
                                points[keypoints[0]][0])

        # Determine if the right wrist is above the right shoulder and
        # if it is,return None
        right_wrist_above = points[keypoints[5]][1] < points[keypoints[1]][1]
        if right_wrist_above:
            return None

        # Determine the angle of the right arm created by the right elbow
        right_elbow_angle = self.calculateAngle(points[keypoints[1]],
                                                points[keypoints[3]],
                                                points[keypoints[5]])

        # Determine the horizontal angle of the right arm
        right_arm_angle = self.calculateHorizontalAngle(points[keypoints[1]],
                                                        points[keypoints[5]])

        # Determine if the right arm is straight [120,180]°
        right_arm_straight = 120 <= right_elbow_angle <= 180

        # Determine if the right arm is horizontal [0,45]°
        right_arm_horizontal = right_arm_angle <= 45

        # If the right arm is both straight and horizontal, return None
        if right_arm_straight and right_arm_horizontal:
            return None

        # Determine the angle of the left arm created by the left elbow and its
        # confidence, otherwise return None
        angle_confidence = 1.0
        if keypoints[2] in points:
            left_elbow_angle = self.calculateAngle(points[keypoints[0]],
                                                   points[keypoints[2]],
                                                   points[keypoints[4]])
            angle_confidence = left_elbow_angle / 180.0
        else:
            return None

        # If the left wrist is above a specific height, determine the
        # confidences and store the data in the data class,
        # otherwise return None
        if left_wrist_height > 0.2:
            height_confidence = min(left_wrist_height / 0.5, 1.0)
            orientation_confidence = 1.0 - min(left_wrist_orient / 0.5, 1.0)
            confidence = (angle_confidence + height_confidence + orientation_confidence) / 3.0
            return poseDetection(
                pose_name='Left Arm Up',
                confidence=min(confidence, 1.0),
                details={
                    'left_wrist_height': left_wrist_height,
                    'left_wrist_orientation': left_wrist_orient,
                }
            )
        else:
            return None

    def rightarm_up(self, points: Dict[int, Tuple[float, float]]) -> Optional[poseDetection]:
        """Detect if a right hand up body pose has been performed

        Args:
            points: Dictionary mapping keypoint index to (x, y) position

        Returns:
            poseDetection object if detected, otherwise None
        """
        # Extract required keypoints from Coco dataset keypoints
        K = self.Keypoints
        keypoints = [K['left-shoulder'], K['right-shoulder'],
                     K['left-elbow'], K['right-elbow'],
                     K['left-wrist'], K['right-wrist']]

        # If not all of the required keypoints are present, return None
        if not all(idx in points for idx in keypoints):
            return None

        # Determine if the right wrist is above the right shoulder
        right_wrist_above = points[keypoints[5]][1] < points[keypoints[1]][1]
        if not right_wrist_above:
            return None

        # Determine the height of the right wrist
        right_wrist_height = points[keypoints[1]][1] - points[keypoints[5]][1]

        # Determine the orientation of the right wrist w.r.t. the right shoulder
        right_wrist_orient = abs(points[keypoints[5]][0] -
                                 points[keypoints[1]][0])

        # Determine if the left wrist is above the left shoulder and
        # if it is, return None
        left_wrist_above = points[keypoints[4]][1] < points[keypoints[0]][1]
        if left_wrist_above:
            return None

        # Determine the angle of the left arm created by the left elbow
        left_elbow_angle = self.calculateAngle(points[keypoints[0]],
                                               points[keypoints[2]],
                                               points[keypoints[4]])

        # Determine the horizontal angle of the left arm
        left_arm_angle = self.calculateHorizontalAngle(points[keypoints[0]],
                                                       points[keypoints[4]])

        # Determine if the left arm is straight [120,180]°
        left_arm_straight = 120 <= left_elbow_angle <= 180

        # Determine if the left arm is horizontal [0,45]°
        left_arm_horizontal = left_arm_angle <= 45

        # If the left arm is both straight and horizontal, return None
        if left_arm_straight and left_arm_horizontal:
            return None

        # Determine the angle of the right arm created by the right elbow and
        # its confidence, otherwise return None
        angle_confidence = 1.0
        if keypoints[3] in points:
            right_elbow_angle = self.calculateAngle(points[keypoints[1]],
                                                    points[keypoints[3]],
                                                    points[keypoints[5]])
            angle_confidence = right_elbow_angle / 180.0
        else:
            return None

        # If the right wrist is above a specific height, determine the
        # confidences and store the data in the data class,
        # otherwise return None
        if right_wrist_height > 0.2:
            height_confidence = min(right_wrist_height / 0.5, 1.0)
            orientation_confidence = 1.0 - min(right_wrist_orient / 0.5, 1.0)
            confidence = (angle_confidence + height_confidence + orientation_confidence) / 3.0
            return poseDetection(
                pose_name='Right Arm Up',
                confidence=min(confidence, 1.0),
                details={
                    'right_wrist_height': right_wrist_height,
                    'right_wrist_orientation': right_wrist_orient
                }
            )
        else:
            return None

    def classifyPose(self, keypoints: List[Tuple[float, float, float]]) -> Optional[poseDetection]:
        """Classify the pose from keypoints

        Args:
            keypoints: List of (x, y, confidence) tuples

        Returns:
            poseObject if detected, otherwise None
        """
        # Parse the  received keypoints
        points = self.parse_keypoints(keypoints)

        # Define the list of poses that can be detected
        detections = [self.tpose_gesture(points),
                      self.handsup_gesture(points),
                      self.leftarmwide_gesture(points),
                      self.rightarmwide_gesture(points),
                      self.leftarm_up(points),
                      self.rightarm_up(points),]

        # If a pose is detected within the specified list, return the pose with
        # the highest confidence
        valid_detection = [d for d in detections if d is not None]
        if valid_detection:
            return max(valid_detection, key=lambda x: x.confidence)

        return None

    def YOLO11classifyPose(self, keypoints):
        if keypoints is None:
            return None

        try:
            if hasattr(keypoints, 'data'):
                keypoints_data = keypoints.data

                if hasattr(keypoints_data, 'cpu'):
                    keypoints_array = keypoints_data.cpu().numpy()
                elif hasattr(keypoints_data, 'numpy'):
                    keypoints_array = keypoints_data.numpy()
                else:
                    keypoints_array = np.array(keypoints_data)
            else:
                return None

            if keypoints_array.size == 0:
                return None

            while len(keypoints_array.shape) > 2 and keypoints_array.shape[0] > 0:
                keypoints_array = keypoints_array[0]

            if keypoints_array.shape != (17, 3):
                return None

            keypoints_list = [(float(x), float(y), float(conf)) for x, y, conf in keypoints_array]

            return self.classifyPose(keypoints_list)

        except Exception:
            return None
