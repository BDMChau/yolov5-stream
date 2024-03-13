module.exports = (tf) => {
  function analyzePose(pose) {
    const bodyParts = pose.keypoints;
    let headY = null,
      feetY = null,
      hipY = null;
    for (const part of bodyParts) {
      switch (part.name) {
        case "nose":
        case "left_eye":
        case "right_eye":
        case "left_ear":
        case "right_ear":
          headY = headY === null ? part.y : Math.min(headY, part.y);
          break;

        case "left_ankle":
        case "right_ankle":
          feetY = feetY === null ? part.y : Math.max(feetY, part.y);
          break;

        case "left_hip":
        case "right_hip":
          hipY = hipY === null ? part.y : Math.max(hipY, part.y);
          break;

        default:
          break;
      }
    }
    if (headY === null || feetY === null || hipY === null) {
      return "Insufficient data";
    }

    // Calculate height
    const personHeight = feetY - headY;

    // Set thresholds based on height
    const fallThreshold = personHeight * 0.7; // This threshold can be adjusted.
    const sitThreshold = personHeight * 0.3; // This threshold can be adjusted.

    if (Math.abs(headY - feetY) < fallThreshold) {
      return "Fallen";
    } else if (Math.abs(headY - hipY) < sitThreshold) {
      return "Sitting";
    } else {
      return "Standing";
    }
  }

  function isPersonFallen(poses) {
    let isTrue = null;
    for (let i = 0; i < poses.length; i++) {
      const pose = poses[i];
      isTrue = isTrue === null ? analyzePose(pose) : isTrue;
      pose.hasFallen = isTrue;
    }
    return isTrue;
  }
  return isPersonFallen;
};
