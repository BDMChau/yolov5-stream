var tf;
const tfjsBuild = process.argv[2];
try {
  switch (tfjsBuild) {
    case "gpu":
      console.log("GPU Test for Tensorflow Module");
      tf = require("@tensorflow/tfjs-node-gpu");
      break;
    case "cpu":
      console.log("CPU Test for Tensorflow Module");
      tf = require("@tensorflow/tfjs-node");
      break;
    default:
      console.log("Nothing selected, using CPU Module for test.");
      console.log(
        `Hint : Run the script like one of the following to specify cpu or gpu.`
      );
      console.log(`node test.js cpu`);
      console.log(`node test.js gpu`);
      tf = require("@tensorflow/tfjs-node-gpu");
      break;
  }
} catch (err) {
  console.log(`Selection Failed. Could not load desired module. ${tfjsBuild}`);
  console.log(err);
}
if (!tf) {
  try {
    tf = require("@tensorflow/tfjs-node-gpu");
  } catch (err) {
    try {
      tf = require("@tensorflow/tfjs-node");
    } catch (err) {
      return console.log("tfjs-node could not be loaded");
    }
  }
}
module.exports = tf;
