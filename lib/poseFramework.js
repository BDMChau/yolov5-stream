const fs = require("fs").promises;
const sharp = require("sharp");
const posesFolder = `${__dirname}/poses/`;

const peopleTags =
  config.peopleTags || new Set(["Person", "Man", "Woman", "Boy", "Girl"]);
const tf = require("./loadTF.js");
const posenet = require("@tensorflow-models/pose-detection");
const model = posenet.SupportedModels.MoveNet;
const detector = await posenet.createDetector(model);
let poseChecks = {};
let poseChecksList = [];
async function getPoseChecks() {
  const checkList = (await fs.readdir(posesFolder)).filter((item) =>
    item.includes(".js")
  );
  const checks = {};
  checkList.forEach((item) => {
    checks[item] = require(`${posesFolder}${item}`)(tf);
    console.log(`Pose Check Loaded! : `, item);
  });
  poseChecks = checks;
  poseChecksList = Object.keys(checks);
}

async function loadImage(buffer) {
  const tensor3d = tf.node.decodeJpeg(buffer, 3);
  return tensor3d;
}
function cropImage(buffer, cropOptions) {
  return new Promise((resolve) => {
    sharp(buffer)
      .metadata()
      .then((metadata) => {
        // Check if crop area is within the image bounds
        const mostRight = cropOptions.left + cropOptions.width;
        const mostDown = cropOptions.top + cropOptions.height;
        if (mostRight > metadata.width || mostDown > metadata.height) {
          let widthMinus = mostRight - metadata.width;
          let heightMinus = mostDown - metadata.height;
          widthMinus = widthMinus > 0 ? widthMinus : 0;
          heightMinus = heightMinus > 0 ? heightMinus : 0;
          cropOptions.width = cropOptions.width - widthMinus;
          cropOptions.height = cropOptions.height - heightMinus;
        }

        // Proceed with cropping if the area is valid
        sharp(buffer)
          .rotate() // Auto-orient based on EXIF
          .extract(cropOptions)
          .toBuffer((err, outputBuffer) => {
            if (err) {
              if (config.debugLog)
                console.log(`Failed to Crop`, err, cropOptions, buffer.length);
              resolve(null);
            } else {
              resolve(outputBuffer);
            }
          });
      })
      .catch((err) => {
        console.log("Error reading image metadata:", err);
        resolve(null);
      });
  });
}

async function detectPose(frame) {
  const imgTensor = await loadImage(frame);
  const poses = await detector.estimatePoses(imgTensor, {
    flipHorizontal: false,
    // maxDetections: 1,
  });
  imgTensor.dispose();
  return poses;
}
function bboxToArrayToObject(bbox) {
  return {
    x: bbox[0],
    y: bbox[1],
    width: bbox[2],
    height: bbox[3],
  };
}
async function detectAllPoses(buffer) {
  let poses = await detectPose(buffer);
  const response = {};
  for (let i = 0; i < poseChecksList.length; i++) {
    const poseToCheck = poseChecksList[i];
    const poseToCheckName = poseToCheck.replace(".js", "");
    const poseCheck = poseChecks[poseToCheck];
    response[poseToCheckName] = await poseCheck(poses);
  }
  return response;
}
async function setPosesToMatrices(theEvent, imageBuffer) {
  const matrices = theEvent.details.matrices;
  const peopleMatrices = matrices.filter((matrix) =>
    peopleTags.has(matrix.tag)
  );
  const otherMatrices = matrices.filter(
    (matrix) => !peopleTags.has(matrix.tag)
  );
  const resultMatrices = [...otherMatrices];
  if (peopleMatrices.length > 0) {
    for (let i = 0; i < peopleMatrices.length; i++) {
      const v = peopleMatrices[i];
      let { x, y, width, height, id } = v;
      if (x < 0) x = 0;
      if (y < 0) y = 0;
      const croppedBuffer = await cropImage(imageBuffer, {
        left: parseInt(x),
        top: parseInt(y),
        width: parseInt(width),
        height: parseInt(height),
      });
      if (!croppedBuffer) continue;
      // if(config.debugLog){
      //     const testFile = `${process.cwd()}/test/person${id}.jpg`;
      //     console.log(`sample written to `,testFile)
      //     await fs.writeFile(testFile,croppedBuffer)
      // }
      const poseStatus = await detectAllPoses(croppedBuffer);
      resultMatrices.push({ ...v, pose: poseStatus });
    }
  }
  return resultMatrices;
}

module.exports = {
  getPoseChecks,
  cropImage,
  detectPose,
  poseChecks,
  poseChecksList,
  setPosesToMatrices,
};
