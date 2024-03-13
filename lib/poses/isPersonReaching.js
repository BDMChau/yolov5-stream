module.exports = (tf) => {
  function analyzePose(bodyParts) {
    let result = {
      left: { pose: "Neutral", confidence: 0 },
      right: { pose: "Neutral", confidence: 0 },
    };

    let leftWristY = null,
      leftHipY = null,
      leftWristX = null,
      leftHipX = null,
      leftWristScore = null,
      leftHipScore = null;
    let rightWristY = null,
      rightHipY = null,
      rightWristX = null,
      rightHipX = null,
      rightWristScore = null,
      rightHipScore = null;

    for (const part of bodyParts) {
      switch (part.name) {
        case "left_wrist":
          leftWristY = part.y;
          leftWristX = part.x;
          leftWristScore = part.score;
          break;
        case "right_wrist":
          rightWristY = part.y;
          rightWristX = part.x;
          rightWristScore = part.score;
          break;
        case "left_hip":
          leftHipY = part.y;
          leftHipX = part.x;
          leftHipScore = part.score;
          break;
        case "right_hip":
          rightHipY = part.y;
          rightHipX = part.x;
          rightHipScore = part.score;
          break;
        default:
          break;
      }
    }

    // Check for left side
    if (leftWristY !== null && leftHipY !== null) {
      result.left.confidence = (leftWristScore + leftHipScore) / 2;

      if (Math.abs(leftWristY - leftHipY) > 100) {
        result.left.pose = "Reaching high";
      } else if (Math.abs(leftWristX - leftHipX) > 100) {
        result.left.pose = "Hand extended";
      }
    } else {
      result.left.pose = "Insufficient data";
    }

    // Check for right side
    if (rightWristY !== null && rightHipY !== null) {
      result.right.confidence = (rightWristScore + rightHipScore) / 2;

      if (Math.abs(rightWristY - rightHipY) > 100) {
        result.right.pose = "Reaching high";
      } else if (Math.abs(rightWristX - rightHipX) > 100) {
        result.right.pose = "Hand extended";
      }
    } else {
      result.right.pose = "Insufficient data";
    }

    return result;
  }

  function isPersonsReaching(poses) {
    let isTrue = false;
    poses.forEach((pose) => {
      const bodyParts = pose.keypoints;
      pose.isReaching = isTrue || analyzePose(bodyParts);
      isTrue = pose.isReaching;
    });
    return isTrue;
  }
  return isPersonsReaching;
};
