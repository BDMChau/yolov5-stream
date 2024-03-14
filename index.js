const express = require("express");
const async = require("async");
const { raptorInference, handleLoadModels } = require("./lib/detector");
const { createCanvas, loadImage } = require("canvas");
const { pipeline, PassThrough } = require("stream");
const util = require("util");
const exec = util.promisify(require("child_process").exec);
const http = require("http");
const { spawn } = require("child_process");
const config = require("./config");

const arrayFFmpegProcess = [];

const configRatio = {
  originalWidth: 640,
  originalHeight: 480,
  resizeWidth: 640,
  resizeHeight: 480,
};

const PORT = 5001;
const app = express();
const httpServer = http.createServer(app);
httpServer.listen({ port: PORT });
console.log(`AiT Camera Stream Server running at http://localhost:${PORT}/`);

const processFramesFromRTSPStream = async (url, queue, imagesStream) => {
  const ffmpegProcess = spawn("ffmpeg", [
    "-re",
    "-rtsp_transport",
    "tcp",
    "-i",
    url,
    "-f",
    "image2pipe",
    "-q:v",
    "8",
    "-vf",
    `scale=${configRatio.originalWidth}:${configRatio.originalHeight},fps=6`,
    "-c:v",
    "mjpeg",
    "-",
  ]);
  arrayFFmpegProcess.push(ffmpegProcess);

  ffmpegProcess.stdout.on("data", async (chunk) => {
    // console.log("chunk", chunk);

    const task = {
      frameBuffer: chunk,
      detectOptions: {
        isObjectDetectionSeparate: false,
      },
    };

    queue.push({
      task,
      imagesStream,
    });
  });

  ffmpegProcess.stderr.on("data", (data) => {
    console.log("ffmpeg stderr:", data.toString());
  });
  ffmpegProcess.on("close", (code) => {
    if (code !== 0) {
      console.log("ffmpeg exited with code", code);
    }
  });
};

const handleImageResult = async (imgResult, imagesStream) => {
  const ffmpegProcessOutput = spawn("ffmpeg", [
    "-i",
    "-",
    "-c:v",
    "mjpeg",
    "-f",
    "mpjpeg",
    "-boundary_tag",
    "raptorvision",
    "-",
  ]);
  ffmpegProcessOutput.stdin.write(imgResult);
  ffmpegProcessOutput.stdin.end();

  ffmpegProcessOutput.stdout.on("data", (chunk) => {
    imagesStream.write(chunk);
  });
};

const worker = async ({ task, imagesStream }) => {
  const currentTime = new Date().getTime();

  let detectResponse = [];
  try {
    detectResponse = await raptorInference(
      task.frameBuffer,
      task.detectOptions
    );
  } catch (error) {
    console.log("raptorInference ERROR:", error);
  }

  // Load the image from buffer
  try {
    const ratioWidth = configRatio.originalWidth / configRatio.resizeWidth;
    const ratioHeight = configRatio.originalHeight / configRatio.resizeHeight;

    const img = await loadImage(task.frameBuffer);
    const canvas = createCanvas(img.width, img.height);
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0, img.width, img.height);

    for (let i = 0; i < detectResponse.length; i++) {
      const { confidence, width, height, x, y, tag, points } =
        detectResponse[i];

      console.log("detectResponse", tag, confidence);

      // Draw the box
      ctx.strokeStyle = "blue";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.rect(
        x * ratioWidth,
        y * ratioHeight,
        width * ratioWidth,
        height * ratioHeight
      );
      ctx.stroke();

      //draw label
      ctx.font = "16px Arial";
      ctx.fillStyle = "blue";
      ctx.fillText(
        `${tag}: ${confidence.toFixed(2)}`,
        x * ratioWidth,
        y * ratioHeight - 5
      );

      if (points?.length > 0) {
        for (let i = 0; i < points.length; i++) {
          const { index, x, y } = points[i];

          ctx.font = "16px Arial";
          ctx.fillStyle = "red";
          ctx.fillText(index, x * ratioWidth, y * ratioHeight);
        }
      }
    }

    // Save the result image
    const imgResult = canvas.toBuffer();
    // const frameFileName = `imgs/frame_${currentTime}.jpg`;
    // fs.writeFileSync(frameFileName, imageBufferResult);

    handleImageResult(imgResult, imagesStream);
  } catch (error) {
    console.log("canvas err", error);
  }
};

app.get(`/get-stream`, async (req, res) => {
  res.writeHead(200, {
    "Content-Type": "multipart/x-mixed-replace;boundary=raptorvision",
    "Cache-Control": "no-cache",
    Connection: "keep-alive",
    Pragma: "no-cache",
  });

  res.on("close", async () => {
    console.log("on close");

    // Reload behavior: the close event will be called when connection is reloaded >> so it will empty the current process instead of previous
    const prevProcess = arrayFFmpegProcess[0];
    if (prevProcess) {
      prevProcess.kill();
      try {
        await exec(
          `kill -9 $(ps -f -C ffmpeg | grep ${prevProcess.pid} | awk '{print $2}')`
        );
      } catch (error) {
        console.log("kill ffmpeg err: ", error);
      }

      arrayFFmpegProcess.shift();
    }
  });

  const videoUrl = "https://cdn.shinobi.video/videos/theif4.mp4";

  let rtspStreamUrl = "";

  // 0: AXIS, 1: AMCREST
  if (config.cameraType === 0) {
    rtspStreamUrl = `rtsp://raptor:Raptor123!@${config.IP}/axis-media/media.amp`;
  } else {
    rtspStreamUrl = `rtsp://raptor:Raptor123!@${config.IP}:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif`;
  }
  console.log(rtspStreamUrl);

  if (!rtspStreamUrl) {
    return res.json("no data");
  }

  const imagesStream = new PassThrough();
  await handleLoadModels(); // get detect model function

  queue = async.queue(worker, 1);
  processFramesFromRTSPStream(rtspStreamUrl, queue, imagesStream);

  pipeline(imagesStream, res, (err) => {
    if (err) {
      console.error("Error piping output to response:", err);
      res.end();
    }
  });
});
