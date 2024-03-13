const axios = require("axios");
const spawn = require("child_process").spawn;
const FormData = require("form-data");
const util = require("util");
const exec = util.promisify(require("child_process").exec);
const config = require("../config");

const weightsPath = `${process.cwd()}/weights/${config.WEIGHT_FILE_NAME}`;
const serverPath = `${process.cwd()}/yolov5/yolov5_flask_server.py`;
const serverPort = 8989;

const flaskUrl = `http://127.0.0.1:${serverPort}`;
const classesFilePath = `${process.cwd()}/weights/classes.txt`;
// const classes = (await fs.readFile(classesFilePath, "utf8")).split("\n");
let modelProcess = null;

async function detect(imageBuffer) {
  try {
    const formData = new FormData();
    formData.append("file", imageBuffer, {
      filename: `${new Date().getTime()}.jpg`,
    });

    const response = await axios({
      method: "POST",
      url: flaskUrl + "/detect",
      data: formData,
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });

    return response.data;
  } catch (error) {
    console.error("Error uploading image:", error.message);
    return [];
  }
}

function loadModel(options = {}) {
  return new Promise(async (resolve) => {
    console.log("Loading Model...");
    options.onData = options.onData || function () {};

    try {
      // kill port if exist
      await exec(`sudo kill -9 $(sudo lsof -t -i:${serverPort})`);
    } catch (error) {
      console.log("KILL SERVER PYTHON ERROR: ", error);
    }

    modelProcess = spawn("python", [
      serverPath,
      "--port",
      serverPort,
      "--weights",
      weightsPath,
    ]);

    let onReady = (data) => {
      const lines = data.toString();
      if (lines.indexOf("Serving Flask app") > -1) {
        console.log("Loaded Model!");
        modelProcess.stdout.off("data", onReady);
        resolve();
      } else if (lines.indexOf("Traceback") > -1) {
        console.error("Failed to Load Model");
        modelProcess.stdout.off("data", onReady);
        resolve();
      }
    };

    modelProcess.stdout.on("data", onReady);
    modelProcess.stdout.on("data", (data) => {
      const lines = data.toString();
      options.onData(lines);
    });

    modelProcess.on("close", (code) => {
      // never let it die
      console.log("Model Died!");
      setTimeout(() => {
        loadModel(options);
      }, 5000);
    });

    modelProcess.stderr.on("data", (data) => {
      console.error(`Detector Server : ${data}`);
    });
  });
}

module.exports = { detect, loadModel };
