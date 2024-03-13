const config = require("./config");

const {
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
  getNoticesForMatrices,
  setMostCommonPosesAndFilterMissing,
  capitalizeMatrixTags,
  setNoticesToMatrices,
  checkForStaleMatrices,
  peopleTags,
} = require(`./tracking.js`);

const detectorBase = config.detectorBase || "yolov5pt";
const currentNotices = {};
const redFlagAttire = config.redFlagAttire || ["Hat", "Sunglasses"];
const redFlagProducts = config.redFlagProducts || ["Bottle", "Tin can"];
const redFlagContainers = config.redFlagContainers || [
  "Suitcase",
  "Backpack",
  "Handbag",
];
const movePercent = parseInt(config.movePercent) || 3;

const acceptedTags = new Set([
  ...redFlagAttire,
  ...redFlagProducts,
  ...redFlagContainers,
  //   ...Array.from(peopleTags),
]);
const redFlagProductAndContainerTags = new Set([
  ...redFlagProducts,
  ...redFlagContainers,
]);

let detect = null;
let loadModel = null;

const { setPosesToMatrices } = await require("../libs/poseFramework.js")(
  config
);
// const dlib = await require('../libs/dlibTracker.js')(config);

function getImageSizeFromEvent(d) {
  var isObjectDetectionSeparate = d.mon.detector_use_detect_object === "1";
  var width = parseFloat(
    isObjectDetectionSeparate && d.mon.detector_scale_y_object
      ? d.mon.detector_scale_y_object
      : d.mon.detector_scale_y
  );
  var height = parseFloat(
    isObjectDetectionSeparate && d.mon.detector_scale_x_object
      ? d.mon.detector_scale_x_object
      : d.mon.detector_scale_x
  );
  return { width, height, isObjectDetectionSeparate };
}

const handleLoadModels = async () => {
  try {
    switch (detectorBase) {
      case "yolov5pt":
        const ptDetect = await require("./yoloObjectDetector.js");
        detect = ptDetect.detect;
        loadModel = ptDetect.loadModel;
        break;
      default:
        break;
    }

    //   await dlib.loadModel();
    if (loadModel) {
      // await loadModel();
    }

    console.log("handleLoadModels OK");
  } catch (error) {
    console.log("handleLoadModels ERROR: ", error);
  }
};

async function raptorInference(
  buffer,
  {
    width,
    height,
    isObjectDetectionSeparate,
    groupKey,
    monitorId,
    monitorDetails,
  }
) {
  const timeStart = new Date();
  const matrices = await detect(buffer);
  const hasMatrices = matrices.length !== 0;
  const theEvent = {
    f: "trigger",
    id: monitorId,
    ke: groupKey,
    details: {
      plug: config.plug,
      name: "pose",
      reason: "object",
      matrices: matrices,
      imgHeight: width,
      imgWidth: height,
      time: new Date() - timeStart,
    },
    frame: buffer,
  };
  const eventDetails = theEvent.details;
  try {
    // sauce start
    if (hasMatrices) {
      eventDetails.matrices = capitalizeMatrixTags(theEvent);
      eventDetails.matrices = trackMatrices(theEvent);
    }

    // eventDetails.matrices = await dlib.track(buffer, theEvent);

    if (eventDetails.matrices.length > 0) {
      eventDetails.matrices = trackMatrices(theEvent);
      eventDetails.matrices = combinePeopleAndMiscObjects(
        eventDetails.matrices,
        width,
        height
      );
      eventDetails.matrices = await setPosesToMatrices(theEvent, buffer);
      addToTrackedHistory(theEvent);
      const { resultMatrices: filteredOnesMissingLotsOfPoses, missingPoses } =
        setMostCommonPosesAndFilterMissing(theEvent);
      if (missingPoses.length > 0) {
        // dlib.clearTrackersByMatrices(groupKey, monitorId, missingPoses);
      }
      eventDetails.matrices = filteredOnesMissingLotsOfPoses;
      const { resultMatrices: notStaleMatrices, staleMatrices } =
        checkForStaleMatrices(theEvent);
      eventDetails.matrices = notStaleMatrices;

      if (staleMatrices.length > 0) {
        // dlib.clearTrackersByMatrices(groupKey, monitorId, staleMatrices);
      }

      markMatricesWithRedFlagTags(theEvent, redFlagProductAndContainerTags);
      eventDetails.matrices = identifyMissingNearbyObjectsOfPeople(
        theEvent,
        peopleTags,
        redFlagProducts
      );
      if (eventDetails.matrices.length > 0) {
        eventDetails.matrices = checkMissingItemsNearContainers(
          theEvent,
          redFlagContainers
        );
      }
      setNoticesToMatrices(theEvent, redFlagProducts, redFlagContainers);
      getNoticesForMatrices(theEvent);
      // sauce end
    }
  } catch (err) {
    console.error("INSIDE raptorInference", err);
  }
  return eventDetails.matrices;
}

module.exports = {
  raptorInference,
  currentNotices,
  getImageSizeFromEvent,
  handleLoadModels,
};
