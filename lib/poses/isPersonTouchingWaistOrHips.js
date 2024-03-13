module.exports = (tf) => {
  function checkHandPlacement(bodyParts) {
    let result = {
      left: "Not detected",
      right: "Not detected",
    };

    let leftWristY = null,
      leftHipY = null,
      leftWristX = null,
      leftHipX = null,
      leftWristScore = null;
    let rightWristY = null,
      rightHipY = null,
      rightWristX = null,
      rightHipX = null,
      rightWristScore = null;

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
          break;
        case "right_hip":
          rightHipY = part.y;
          rightHipX = part.x;
          break;
        default:
          break;
      }
    }

    const yThreshold = 20; // Threshold for y-direction (can be adjusted)
    const xThreshold = 20; // Threshold for x-direction (can be adjusted)
    const visibilityThreshold = 0.25; // Threshold for visibility based on score (can be adjusted)

    // Check for left side
    if (leftWristScore !== null && leftWristScore < visibilityThreshold) {
      result.left = "Not visible";
    } else if (
      leftWristY !== null &&
      leftHipY !== null &&
      leftWristX !== null &&
      leftHipX !== null
    ) {
      if (
        Math.abs(leftWristY - leftHipY) <= yThreshold &&
        Math.abs(leftWristX - leftHipX) <= xThreshold
      ) {
        result.left = "Near hips";
      } else if (
        leftWristY < leftHipY &&
        leftHipY - leftWristY <= yThreshold &&
        Math.abs(leftWristX - leftHipX) <= xThreshold
      ) {
        result.left = "Over belly/waistline";
      }
    }

    // Check for right side
    if (rightWristScore !== null && rightWristScore < visibilityThreshold) {
      result.right = "Not visible";
    } else if (
      rightWristY !== null &&
      rightHipY !== null &&
      rightWristX !== null &&
      rightHipX !== null
    ) {
      if (
        Math.abs(rightWristY - rightHipY) <= yThreshold &&
        Math.abs(rightWristX - rightHipX) <= xThreshold
      ) {
        result.right = "Near hips";
      } else if (
        rightWristY < rightHipY &&
        rightHipY - rightWristY <= yThreshold &&
        Math.abs(rightWristX - rightHipX) <= xThreshold
      ) {
        result.right = "Over belly/waistline";
      }
    }

    return result;
  }

  function isPersonsReaching(poses) {
    let isTrue = false;
    poses.forEach((pose) => {
      const bodyParts = pose.keypoints;
      pose.isPersonTouchingWaistOrHips =
        isTrue || checkHandPlacement(bodyParts);
      isTrue = pose.isPersonTouchingWaistOrHips;
    });
    return isTrue;
  }
  return isPersonsReaching;
};
