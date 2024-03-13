const movingThings = require("shinobi-node-moving-things-tracker").Tracker;

const config = require("../config");

const objectTrackers = {};
const objectTrackerTimeouts = {};
const peopleTags = new Set(
  config.peopleTags || ["person", "Person", "Man", "Woman", "Boy", "Girl"]
);
const trackedMatrices = {};
const trackedNotices = {};
const trackedNoticesTimeout = {};
const trackerTimeoutPeriod = 1000 * 60;
const trackerSampleSize = 30;
if (config.debugLog === true) {
  function debugLog(...args) {
    console.log(...args);
  }
} else {
  function debugLog() {}
}
function shiftSet(set) {
  for (const res of set) {
    set.delete(res);
    return res;
  }
}
function resetObjectTracker(trackerId, matrices) {
  const Tracker = movingThings.newTracker();
  objectTrackers[trackerId] = {
    frameCount: 1,
    tracker: Tracker,
    lastPositions: [],
  };
  return objectTrackers[trackerId];
}
function setLastTracked(trackerId, trackedMatrices) {
  const theTracker = objectTrackers[trackerId];
  theTracker.lastPositions = trackedMatrices;
}
function getTracked(trackerId) {
  const theTracker = objectTrackers[trackerId];
  const frameCount = theTracker.frameCount;
  const trackedObjects = theTracker.tracker
    .getJSONOfTrackedItems()
    .map((matrix) => {
      return {
        id: matrix.id,
        tag: matrix.name,
        x: matrix.x,
        y: matrix.y,
        width: matrix.w,
        height: matrix.h,
        confidence: matrix.confidence,
        isZombie: matrix.isZombie,
      };
    });
  return trackedObjects;
}
function trackObject(trackerId, matrices) {
  if (!objectTrackers[trackerId]) {
    resetObjectTracker(trackerId);
  }
  const mappedMatrices = matrices.map((matrix) => {
    return {
      x: matrix.x,
      y: matrix.y,
      w: matrix.width,
      h: matrix.height,
      confidence: matrix.confidence,
      name: matrix.tag,
    };
  });
  const theTracker = objectTrackers[trackerId];
  theTracker.tracker.updateTrackedItemsWithNewFrame(
    mappedMatrices,
    theTracker.frameCount
  );
  ++theTracker.frameCount;
}
function trackObjectWithTimeout(trackerId, matrices) {
  clearTimeout(objectTrackerTimeouts[trackerId]);
  objectTrackerTimeouts[trackerId] = setTimeout(() => {
    objectTrackers[trackerId].tracker.reset();
    delete objectTrackers[trackerId];
    delete objectTrackerTimeouts[trackerId];
  }, trackerTimeoutPeriod);
  trackObject(trackerId, matrices);
}
function objectHasMoved(matrices, options = {}) {
  const { imgHeight = 1, imgWidth = 1, threshold = 0 } = options;
  for (let i = 0; i < matrices.length; i++) {
    const current = matrices[i];
    if (i < matrices.length - 1) {
      const next = matrices[i + 1];
      let totalDistanceMoved = 0;
      let numPointsCompared = 0;
      if (next) {
        // Compare each corner of the matrices
        const currentCorners = [
          { x: current.x, y: current.y },
          { x: current.x + current.width, y: current.y },
          { x: current.x, y: current.y + current.height },
          { x: current.x + current.width, y: current.y + current.height },
        ];
        const nextCorners = [
          { x: next.x, y: next.y },
          { x: next.x + next.width, y: next.y },
          { x: next.x, y: next.y + next.height },
          { x: next.x + next.width, y: next.y + next.height },
        ];
        for (let j = 0; j < currentCorners.length; j++) {
          const currentCorner = currentCorners[j];
          const nextCorner = nextCorners[j];
          const dx = nextCorner.x - currentCorner.x;
          const dy = nextCorner.y - currentCorner.y;
          const distanceMoved = Math.sqrt(Math.pow(dx, 2) + Math.pow(dy, 2));
          const distanceMovedPercent =
            (100 * distanceMoved) / Math.max(current.width, current.height);
          totalDistanceMoved += distanceMovedPercent;
          numPointsCompared++;
        }
        const averageDistanceMoved = totalDistanceMoved / numPointsCompared;
        if (averageDistanceMoved < threshold) {
          continue;
        } else {
          return true;
        }
      }
    }
  }
  return false;
}
function groupMatricesById(matrices) {
  const matrixById = {};
  const matrixTags = {};

  matrices.forEach((matrix) => {
    const id = matrix.id;
    const tag = matrix.tag;
    if (!matrixById[id]) {
      matrixById[id] = [];
    }
    matrixTags[tag] = id;
    matrixById[id].push(matrix);
  });

  return matrixById;
}
function filterAllMatricesNotMoved({
  trackerIdPrefix,
  imgWidth,
  imgHeight,
  threshold,
  matrices,
}) {
  let peopleMatrices = matrices.filter((matrix) => peopleTags.has(matrix.tag));
  let otherMatrices = matrices.filter((matrix) => !peopleTags.has(matrix.tag));
  let movedMatrices = [...peopleMatrices];

  otherMatrices.forEach((matrix) => {
    const isNearPeople =
      matrix.associatedPeople && matrix.associatedPeople.length > 0;
    if (isNearPeople) {
      const trackerId = `${trackerIdPrefix}${matrix.id}${matrix.tag}`;
      const trackedMatrixSet = trackedMatrices[trackerId];
      if (trackedMatrixSet && trackedMatrixSet.size > 1) {
        const hasMoved = objectHasMoved(Array.from(trackedMatrixSet), {
          imgWidth,
          imgHeight,
          threshold,
        });
        if (hasMoved) {
          movedMatrices.push(matrix);
        }
      }
    }
  });

  return movedMatrices;
}
function isMatrixNearAnother(firstMatrix, secondMatrix, imgWidth, imgHeight) {
  // Calculate the distance between two rectangles
  function rectDistance(rect1, rect2) {
    const xDist = Math.max(
      rect1.x - (rect2.x + rect2.width),
      rect2.x - (rect1.x + rect1.width),
      0
    );
    const yDist = Math.max(
      rect1.y - (rect2.y + rect2.height),
      rect2.y - (rect1.y + rect1.height),
      0
    );
    return Math.sqrt(xDist * xDist + yDist * yDist);
  }

  // Calculate the overlap area
  function overlapArea(rect1, rect2) {
    const xOverlap = Math.max(
      0,
      Math.min(rect1.x + rect1.width, rect2.x + rect2.width) -
        Math.max(rect1.x, rect2.x)
    );
    const yOverlap = Math.max(
      0,
      Math.min(rect1.y + rect1.height, rect2.y + rect2.height) -
        Math.max(rect1.y, rect2.y)
    );
    return xOverlap * yOverlap;
  }

  const pxDistance = rectDistance(firstMatrix, secondMatrix);
  const overlapAreaValue = overlapArea(firstMatrix, secondMatrix);
  const totalArea =
    firstMatrix.width * firstMatrix.height +
    secondMatrix.width * secondMatrix.height -
    overlapAreaValue;
  const overlapPercent =
    totalArea > 0 ? (overlapAreaValue / totalArea) * 100 : 0;
  const distancePercent =
    Math.sqrt(
      Math.pow(pxDistance / imgWidth, 2) + Math.pow(pxDistance / imgHeight, 2)
    ) * 100;
  const isOverlap = overlapAreaValue > 0;
  const nearThreshold = 50;
  const isNear = distancePercent < 5;

  return { pxDistance, overlapPercent, distancePercent, isOverlap, isNear };
}
function combinePeopleAndMiscObjects(matrices, imgWidth, imgHeight) {
  const peopleMatrices = [];
  const otherMatrices = [];
  matrices.forEach((matrix) => {
    if (peopleTags.has(matrix.tag)) {
      peopleMatrices.push({ ...matrix, nearBy: [] });
    } else {
      otherMatrices.push({ ...matrix, associatedPeople: [], color: "green" }); // Initialize associatedPeople array
    }
  });
  const resultMatrices = [...peopleMatrices];
  peopleMatrices.forEach((personMatrix) => {
    otherMatrices.forEach((otherMatrix) => {
      const comparisonResult = isMatrixNearAnother(
        personMatrix,
        otherMatrix,
        imgWidth,
        imgHeight
      );
      // console.error(`comparisonResult (${comparisonResult.overlapPercent}%) : ${otherMatrix.tag} (${otherMatrix.id}) is on ${personMatrix.tag} (${personMatrix.id}) about ${comparisonResult.overlapPercent}%`)
      // console.log('comparisonResult',comparisonResult)
      if (comparisonResult.overlapPercent > 15 || comparisonResult.isNear) {
        personMatrix.nearBy.push({
          ...otherMatrix,
          ...comparisonResult,
        });
        otherMatrix.associatedPeople.push(personMatrix.id);
        otherMatrix.color = "yellow";
        resultMatrices.push(otherMatrix);
      }
    });
  });
  return resultMatrices;
}
function addToTrackedHistory(theEvent) {
  const groupKey = theEvent.ke;
  const monitorId = theEvent.id;
  const matrices = theEvent.details.matrices;
  matrices.forEach((matrix) => {
    const trackerId = `${groupKey}${monitorId}${matrix.id}${matrix.tag}`;
    if (!trackedMatrices[trackerId]) trackedMatrices[trackerId] = new Set();
    trackedMatrices[trackerId].add(matrix);
    if (trackedMatrices[trackerId].size > trackerSampleSize) {
      shiftSet(trackedMatrices[trackerId]);
    }
  });
}
function filterOutLessSeenNearBy(theEvent) {
  const groupKey = theEvent.ke;
  const monitorId = theEvent.id;
  const matrices = theEvent.details.matrices;
  matrices.forEach((matrix) => {
    if (!matrix.nearBy) matrix.nearBy = [];
    const trackerId = `${groupKey}${monitorId}${matrix.id}${matrix.tag}`;
    const trackedSet = trackedMatrices[trackerId];
    if (trackedSet && trackedSet.size > 0) {
      const frequencyMap = new Map();
      trackedSet.forEach((trackedMatrix) => {
        trackedMatrix.nearBy.forEach((nearByMatrix) => {
          const key = JSON.stringify(nearByMatrix); // Assuming 'nearByMatrix' is an object
          frequencyMap.set(key, (frequencyMap.get(key) || 0) + 1);
        });
      });
      matrix.nearBy = matrix.nearBy.filter((nearByMatrix) => {
        const key = JSON.stringify(nearByMatrix);
        return frequencyMap.get(key) / trackedSet.size >= 0.8;
      });
    }
  });
  return theEvent;
}
function separateMatricesByTag(matrices) {
  const groupedByTag = matrices.reduce((acc, matrix) => {
    if (!acc[matrix.tag]) {
      acc[matrix.tag] = [];
    }
    acc[matrix.tag].push(matrix);
    return acc;
  }, {});
  return Object.values(groupedByTag);
}
function trackMatrices(theEvent) {
  const matrices = theEvent.details.matrices;
  const trackerIdPrefix = `${theEvent.ke}${theEvent.id}`;
  const trackedObjects = [];
  separateMatricesByTag(matrices).forEach((matrices) => {
    if (!matrices[0]) return;
    const matrixTag = matrices[0].tag;
    const trackerId = `${trackerIdPrefix}${matrixTag}`;
    trackObjectWithTimeout(trackerId, matrices);
    trackedObjects.push(...getTracked(trackerId));
    setLastTracked(trackerId, trackedObjects);
  });
  return trackedObjects;
}
function markMatricesWithRedFlagTags(theEvent, redFlags) {
  const groupKey = theEvent.ke;
  const monitorId = theEvent.id;
  const matrices = theEvent.details.matrices;

  matrices.forEach((matrix) => {
    const trackerId = `${groupKey}${monitorId}${matrix.id}${matrix.tag}`;
    const trackedMatrixSet = trackedMatrices[trackerId];

    if (trackedMatrixSet) {
      let redFlagCount = 0; // Counter for matrices with red flag tags

      trackedMatrixSet.forEach((trackedMatrix) => {
        // Check if any nearBy matrix has a tag that matches the red flags
        if (
          trackedMatrix.nearBy &&
          trackedMatrix.nearBy.some((nearByMatrix) =>
            redFlags.has(nearByMatrix.tag)
          )
        ) {
          redFlagCount++; // Increment counter for each match
        }
      });

      // Calculate if the red flag count is at least 30% of the trackedMatrixSet
      if (redFlagCount / trackedMatrixSet.size >= 0.3) {
        matrix.suspect = true; // Mark the matrix as suspect
      }
    }
  });
}

function identifyMissingNearbyObjectsOfPeople(
  theEvent,
  peopleTags,
  redFlagProducts
) {
  const matrices = theEvent.details.matrices;
  const peopleMatrices = matrices
    .filter((matrix) => peopleTags.has(matrix.tag))
    .map((matrix) => ({
      ...matrix,
      missingRecently: [],
    }));
  const otherMatrices = matrices.filter(
    (matrix) => !peopleTags.has(matrix.tag)
  );
  peopleMatrices.forEach((personMatrix) => {
    const trackerId = `${theEvent.ke}${theEvent.id}${personMatrix.id}${personMatrix.tag}`;
    const trackedMatrixSet = trackedMatrices[trackerId];
    if (trackedMatrixSet) {
      let nearByFrequencies = {};
      trackedMatrixSet.forEach((trackedMatrix) => {
        trackedMatrix.nearBy.forEach((nearByMatrix) => {
          const nearById = `${nearByMatrix.id}-${nearByMatrix.tag}`;
          if (!nearByFrequencies[nearById]) {
            nearByFrequencies[nearById] = { count: 0, matrix: nearByMatrix };
          }
          nearByFrequencies[nearById].count += 1;
        });
      });
      Object.values(nearByFrequencies).forEach(({ count, matrix }) => {
        if (count / trackedMatrixSet.size >= 0.2) {
          const isMissingFromEvent = !theEvent.details.matrices.some(
            (eventMatrix) =>
              eventMatrix.id === matrix.id && eventMatrix.tag === matrix.tag
          );
          const isRedFlagProduct = redFlagProducts.includes(matrix.tag);
          if (isMissingFromEvent && isRedFlagProduct) {
            personMatrix.missingRecently.push(matrix);
          }
        }
      });
    }
  });

  return peopleMatrices.concat(otherMatrices); //.filter(matrix => matrix.missingRecently.length > 0);
}
function makeNoticeFromMatrix(matrix, redFlagProducts, redFlagContainers) {
  const isThings = [];
  const isSuspect = matrix.suspect === true;
  if (isSuspect) {
    const nearByProducts = matrix.nearBy
      .filter((item) => redFlagProducts.indexOf(item.tag) > -1)
      .map((item) => item.tag);
    const nearByBadContainers = matrix.nearBy
      .filter((item) => redFlagContainers.includes(item.tag))
      .map((item) => item.tag);
    // if(matrix.nearBy.length > 0)console.log(matrix.nearBy)
    let isReaching = false;
    if (matrix.pose) {
      const reaching = matrix.pose.isPersonReaching;
      if (reaching) {
        const isLeftHandReaching =
          reaching.left.pose === "Reaching high" ||
          reaching.left.pose === "Hand extended";
        const isRightHandReaching =
          reaching.right.pose === "Reaching high" ||
          reaching.right.pose === "Hand extended";
        isReaching = isLeftHandReaching || isRightHandReaching;
      }
    }
    let notice = `${matrix.tag} touching ${nearByBadContainers.join(",")}`;
    if (isReaching) {
      notice += ` and is reaching`;
    }
    return notice;
  } else {
    return null;
  }
}
function setNoticeToMatrixTagAndId(
  matrix,
  groupKey,
  monitorId,
  redFlagProducts,
  redFlagContainers
) {
  const matrixId = matrix.id;
  const tag = matrix.tag;
  const trackedId = `${groupKey}${monitorId}${matrixId}${tag}`;
  const notice = makeNoticeFromMatrix(
    matrix,
    redFlagProducts,
    redFlagContainers
  );
  if (!notice) return;
  trackedNotices[trackedId] = notice;
  clearTimeout(trackedNoticesTimeout[trackedId]);
  trackedNoticesTimeout[trackedId] = setTimeout(() => {
    delete trackedNotices[trackedId];
  }, 60000);
}
function setNoticesToMatrices(theEvent, redFlagProducts, redFlagContainers) {
  const groupKey = theEvent.ke;
  const monitorId = theEvent.id;
  const eventMatrices = theEvent.details.matrices;
  eventMatrices.forEach((matrix) => {
    setNoticeToMatrixTagAndId(
      matrix,
      groupKey,
      monitorId,
      redFlagProducts,
      redFlagContainers
    );
  });
}
function getNoticeFromMatrixTagAndId(matrix, groupKey, monitorId) {
  const matrixId = matrix.id;
  const tag = matrix.tag;
  const trackedId = `${groupKey}${monitorId}${matrixId}${tag}`;
  return trackedNotices[trackedId];
}
function getNoticesForMatrices(theEvent) {
  const groupKey = theEvent.ke;
  const monitorId = theEvent.id;
  const eventMatrices = theEvent.details.matrices;
  eventMatrices.forEach((matrix) => {
    const matrixId = matrix.id;
    const tag = matrix.tag;
    const trackedId = `${groupKey}${monitorId}${matrixId}${tag}`;
    const notice = trackedNotices[trackedId];
    if (notice) matrix.notice = notice;
  });
}
function checkMissingItemsNearContainers(theEvent, containerTags) {
  const peopleMatrices = theEvent.details.matrices.filter((matrix) =>
    peopleTags.has(matrix.tag)
  );
  const otherMatrices = theEvent.details.matrices.filter(
    (matrix) => !peopleTags.has(matrix.tag)
  );
  peopleMatrices.forEach((personMatrix) => {
    personMatrix.missingNear = [];
    if (
      personMatrix.missingRecently &&
      personMatrix.missingRecently.length > 0
    ) {
      const trackerId = `${theEvent.ke}${theEvent.id}${personMatrix.id}${personMatrix.tag}`;
      const trackedMatrixSet = trackedMatrices[trackerId];

      if (trackedMatrixSet) {
        personMatrix.missingRecently.forEach((missingItem) => {
          trackedMatrixSet.forEach((trackedMatrix) => {
            if (containerTags.includes(trackedMatrix.tag)) {
              const comparisonResult = isMatrixNearAnother(
                missingItem,
                trackedMatrix,
                theEvent.imgWidth,
                theEvent.imgHeight
              );
              if (comparisonResult.overlapPercent > 5) {
                const missingItemWithDetails = {
                  ...missingItem,
                  missedNear: trackedMatrix,
                };
                personMatrix.missingNear.push(missingItemWithDetails);
                // debugLog(`Missing item ${missingItem.tag} (${missingItem.id}) was near ${trackedMatrix.tag} (${trackedMatrix.id}) for person ${personMatrix.id}`);
              }
            }
          });
        });
      }
    }
  });
  return [...peopleMatrices, ...otherMatrices];
}
function setMostCommonPosesAndFilterMissing(theEvent) {
  const matrices = theEvent.details.matrices;
  const peopleMatrices = matrices.filter((matrix) =>
    peopleTags.has(matrix.tag)
  );
  const otherMatrices = matrices.filter(
    (matrix) => !peopleTags.has(matrix.tag)
  );
  const resultMatrices = [...otherMatrices];
  const missingPoses = [];
  peopleMatrices.forEach((personMatrix) => {
    const trackerId = `${theEvent.ke}${theEvent.id}${personMatrix.id}${personMatrix.tag}`;
    const trackedMatrixSet = trackedMatrices[trackerId];
    if (trackedMatrixSet) {
      const trackedMatrixSetSize = trackedMatrixSet.size;
      const trackSizeGood = trackedMatrixSetSize >= 25;
      if (trackSizeGood) {
        const missingMax = trackedMatrixSetSize * 0.8;
        let leftPoseCounts = {};
        let rightPoseCounts = {};
        let missingPose = 0;
        trackedMatrixSet.forEach((trackedMatrix) => {
          if (!trackedMatrix.pose.isPersonReaching) {
            ++missingPose;
          } else {
            const leftPose = trackedMatrix.pose.isPersonReaching.left.pose;
            const rightPose = trackedMatrix.pose.isPersonReaching.right.pose;
            leftPoseCounts[leftPose] = (leftPoseCounts[leftPose] || 0) + 1;
            rightPoseCounts[rightPose] = (rightPoseCounts[rightPose] || 0) + 1;
          }
        });
        const leftPoseCountsKeys = Object.keys(leftPoseCounts);
        if (missingPose >= missingMax) {
          missingPoses.push(personMatrix);
        } else if (leftPoseCountsKeys.length > 0) {
          const mostCommonLeftPose = leftPoseCountsKeys.reduce((a, b) =>
            leftPoseCounts[a] > leftPoseCounts[b] ? a : b
          );
          const mostCommonRightPose = Object.keys(rightPoseCounts).reduce(
            (a, b) => (rightPoseCounts[a] > rightPoseCounts[b] ? a : b)
          );
          if (!personMatrix.commonPose) {
            personMatrix.commonPose = {
              isPersonReaching: { left: {}, right: {} },
            };
          }
          personMatrix.commonPose.isPersonReaching.left.pose =
            mostCommonLeftPose;
          personMatrix.commonPose.isPersonReaching.right.pose =
            mostCommonRightPose;
          resultMatrices.push(personMatrix);
        } else {
          resultMatrices.push(personMatrix);
        }
      }
    }
  });
  return { resultMatrices, missingPoses };
}
function capitalizeFirstLetter(string) {
  return string.charAt(0).toUpperCase() + string.slice(1);
}
function capitalizeMatrixTags(theEvent) {
  return theEvent.details.matrices.map((matrix) => {
    matrix.tag = capitalizeFirstLetter(matrix.tag);
    return matrix;
  });
}
function checkForStaleMatrices(theEvent) {
  let resultMatrices = [];
  let staleMatrices = [];

  theEvent.details.matrices.forEach((matrix) => {
    const trackerId = `${theEvent.ke}${theEvent.id}${matrix.id}${matrix.tag}`;
    const trackedMatrixSet = trackedMatrices[trackerId];
    if (trackedMatrixSet && trackedMatrixSet.size > 20) {
      const someItems = Array.from(trackedMatrixSet).slice(-10);
      const isStale = someItems.filter((trackedMatrix) => {
        return (
          trackedMatrix.x === matrix.x &&
          trackedMatrix.y === matrix.y &&
          trackedMatrix.width === matrix.width &&
          trackedMatrix.height === matrix.height
        );
      });
      if (someItems.length === isStale.length) {
        staleMatrices.push(matrix);
      } else {
        resultMatrices.push(matrix);
      }
    } else {
      resultMatrices.push(matrix);
    }
  });

  return { resultMatrices, staleMatrices };
}

module.exports = {
  trackObjectWithTimeout,
  resetObjectTracker,
  trackObject,
  getTracked,
  setLastTracked,
  filterAllMatricesNotMoved,
  isMatrixNearAnother,
  combinePeopleAndMiscObjects,
  filterOutLessSeenNearBy,
  separateMatricesByTag,
  addToTrackedHistory,
  trackMatrices,
  markMatricesWithRedFlagTags,
  identifyMissingNearbyObjectsOfPeople,
  checkMissingItemsNearContainers,
  setNoticeToMatrixTagAndId,
  getNoticeFromMatrixTagAndId,
  getNoticesForMatrices,
  setMostCommonPosesAndFilterMissing,
  setNoticesToMatrices,
  peopleTags,
  capitalizeMatrixTags,
  checkForStaleMatrices,
};
