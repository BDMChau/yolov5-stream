import tensorflow as tf
import keras

BODY_PARTS = {
    'NOSE': 0,
    'LEFT_EYE': 1,
    'RIGHT_EYE': 2,
    'LEFT_EAR': 3,
    'RIGHT_EAR': 4,
    'LEFT_SHOULDER': 5,
    'RIGHT_SHOULDER': 6,
    'LEFT_ELBOW': 7,
    'RIGHT_ELBOW': 8,
    'LEFT_WRIST': 9,
    'RIGHT_WRIST': 10,
    'LEFT_HIP': 11,
    'RIGHT_HIP': 12,
    'LEFT_KNEE': 13,
    'RIGHT_KNEE': 14,
    'LEFT_ANKLE': 15,
    'RIGHT_ANKLE': 16
}


def get_center_point(landmarks, leftIndex, rightIndex):  
  left = tf.gather(landmarks, leftIndex, axis=1)
  right = tf.gather(landmarks, rightIndex, axis=1)
  center = left * 0.5 + right * 0.5
    
  return center


def get_pose_size(landmarks, torso_size_multiplier=2.5):
  """Calculates pose size.

  It is the maximum of two values:
    * Torso size multiplied by `torso_size_multiplier`
    * Maximum distance from pose center to any pose landmark
  """
  
  hips_center = get_center_point(landmarks, BODY_PARTS['LEFT_HIP'], 
                                 BODY_PARTS['RIGHT_HIP'])


  shoulders_center = get_center_point(landmarks, BODY_PARTS['LEFT_SHOULDER'],
                                      BODY_PARTS['RIGHT_SHOULDER'])


  torso_size = tf.linalg.norm(shoulders_center - hips_center)

  pose_center_new = get_center_point(landmarks, BODY_PARTS['LEFT_HIP'], 
                                     BODY_PARTS['RIGHT_HIP'])
  pose_center_new = tf.expand_dims(pose_center_new, axis=1)

  pose_center_new = tf.broadcast_to(pose_center_new,
                                    [tf.size(landmarks) // (17*2), 17, 2])

  d = tf.gather(landmarks - pose_center_new, 0, axis=0,
                name="dist_to_pose_center")
  max_dist = tf.reduce_max(tf.linalg.norm(d, axis=0))

  pose_size = tf.maximum(torso_size * torso_size_multiplier, max_dist)

  return pose_size


def normalize_pose_landmarks(landmarks):
  """Normalizes the landmarks translation by moving the pose center to (0,0) and
  scaling it to a constant pose size.
  """
  pose_center = get_center_point(landmarks, BODY_PARTS['LEFT_HIP'], 
                                 BODY_PARTS['RIGHT_HIP'])
  pose_center = tf.expand_dims(pose_center, axis=1)

  pose_center = tf.broadcast_to(pose_center, 
                                [tf.size(landmarks) // (17*2), 17, 2])
  landmarks = landmarks - pose_center

  pose_size = get_pose_size(landmarks)
  landmarks /= pose_size

  return landmarks


def landmarks_to_embedding(landmarks):
  # Flatten the normalized landmark coordinates into a vector
  embedding = keras.layers.Flatten()(landmarks)

  return embedding
