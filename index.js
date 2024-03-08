const { spawn } = require("child_process");
const fs = require("fs");
const async = require("async");
const { raptorInference, handleLoadModels } = require("./detector");
const { createCanvas, loadImage } = require("canvas");

const processFramesFromVideo = async (url, queue) => {
  return new Promise(async (resolve, reject) => {
    let i = 0;
    let width = 1280;
    let height = 720;
    const ffmpeg = spawn("ffmpeg", [
      "-re",
      "-i",
      url,
      "-f",
      "image2pipe",
      "-vf",
      "scale=640:480,fps=10", // Extract 10 frame per second (you can adjust this)
      "-c:v",
      "mjpeg",
      "-",
    ]);

    ffmpeg.stdout.on("data", async (chunk) => {
      let frameData = [];

      if (
        chunk[chunk.length - 2] === 0xff &&
        chunk[chunk.length - 1] === 0xd9
      ) {
        frameData.push(chunk);
        try {
          i++;
          const task = {
            frameBuffer: Buffer.concat(frameData),
            i,
            detectOptions: {
              width,
              height,
              isObjectDetectionSeparate: false,
            },
          };
          queue.push(task);
        } catch (err) {
          console.log(err);
        }
      }
    });

    ffmpeg.on("close", (code) => {
      if (code !== 0) {
        reject(new Error(`ffmpeg exited with code ${code}`));
      } else {
        resolve();
      }
    });
  });
};

const worker = async (task, callback) => {
  const currentTime = new Date().getTime();

  try {
    const detectResponse = await raptorInference(
      task.frameBuffer,
      task.detectOptions
    );
    console.log("detectResponse", detectResponse.length);

    if (detectResponse?.length > 0) {
      // Load the image from buffer
      const img = await loadImage(task.frameBuffer);
      const canvas = createCanvas();
      const ctx = canvas.getContext("2d");
      canvas.width = img.width;
      canvas.height = img.height;
      ctx.drawImage(img, 0, 0, img.width, img.height);

      for (let i = 0; i < detectResponse.length; i++) {
        const { confidence, width, height, x, y, tag } = detectResponse[i];

        // Draw the box
        ctx.strokeStyle = "blue";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.rect(x, y, width, height);
        ctx.stroke();

        //draw label
        ctx.font = "16px Arial";
        ctx.fillStyle = "blue";
        ctx.fillText(`${tag}: ${confidence.toFixed(2)}`, x, y - 5);
      }

      // Save the result image
      const imageData = canvas.toBuffer();
      const frameFileName = `imgs/frame_${currentTime}.jpg`;
      fs.writeFileSync(frameFileName, imageData);
    }
  } catch (error) {
    console.log("worker ERROR:", error);
  }
};

const index = async () => {
  await handleLoadModels();

  const videoUrl = "https://cdn.shinobi.video/videos/theif4.mp4";
  // const videoUrl =
  //   "http://192.168.100.252:8989/get-stream/cWEtc2l0ZQ--/cnRzcDovL3JhcHRvcjpSYXB0b3IxMjMhQDE5Mi4xNjguMTAwLjEzMjo1NTQvY2FtL3JlYWxtb25pdG9yP2NoYW5uZWw9MSZzdWJ0eXBlPTAmdW5pY2FzdD10cnVlJnByb3RvPU9udmlm";

  const queue = async.queue(worker, 1);
  processFramesFromVideo(videoUrl, queue);
};
index();
