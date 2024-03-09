const { spawn } = require("child_process");
const fs = require("fs");
const express = require("express");
const async = require("async");
// const { raptorInference, handleLoadModels } = require("./detector");
const { createCanvas, loadImage } = require("canvas");
const { pipeline, Readable, PassThrough } = require("stream");
const http = require("http");
const child_process = require("child_process");

const PORT = 5000;
const app = express();
const httpServer = http.createServer(app);
httpServer.listen({ port: PORT });
console.log(`AiT Camera Stream Server running at http://localhost:${PORT}/`);

const processFramesFromRTSPStream = async (url, queue, imagesStream, res) => {
  let i = 0;
  let width = 1280;
  let height = 720;

  const ffmpeg = spawn("ffmpeg", [
    "-re",
    "-i",
    url,
    "-vf",
    "scale=640:480,fps=10", // Extract 10 frame per second (you can adjust this)
    "-c:v",
    "mjpeg",
    "-f",
    "mpjpeg",
    "-boundary_tag",
    "raptorvision",
    "-",
  ]);

  // const ffmpeg = spawn("ffmpeg", [
  //   "-re",
  //   "-rtsp_transport",
  //   "tcp",
  //   "-i",
  //   url,
  //   "-f",
  //   "image2pipe",
  //   "-vf",
  //   "scale=640:480,fps=10", // Extract 10 frame per second (you can adjust this)
  //   "-c:v",
  //   "mjpeg",
  //   "-",
  // ]);
  //   pipeline(ffmpeg.stdout, res, (err) => {
  //     if (err) {
  //       console.error("Error piping ffmpeg output to response:", err);
  //       res.end();
  //     } else {
  //       console.log("pipe ok");
  //     }
  //   });

  ffmpeg.stdout.on("data", async (chunk) => {
    let frameData = [];
    // console.log("chunk", chunk);
    const buffer = Buffer.from(chunk, "utf-8");

    // const frameFileName = `imgs/frame_${Date.now()}.jpg`;
    // fs.writeFileSync(frameFileName, chunk);

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

      queue.push({
        task,
        callback: (imgBuffer) => {
          handleCallback(imgBuffer, imagesStream);
        },
      });
    } catch (err) {
      console.log(err);
    }
  });

  ffmpeg.stderr.on("data", (data) => {
    // console.error("ffmpeg error:", data.toString());
  });

  ffmpeg.on("close", (code) => {
    if (code !== 0) {
      console.log("ffmpeg exited with code", code);
    }
  });
};

const handleCallback = (imgResult, imagesStream) => {
  imagesStream.push(imgResult);
};

const worker = async ({ task, callback }) => {
  const currentTime = new Date().getTime();

  let detectResponse = [];
  //   try {
  //     detectResponse = await raptorInference(
  //       task.frameBuffer,
  //       task.detectOptions
  //     );
  //   } catch (error) {
  //     console.log("raptorInference ERROR:", error);
  //   }

  // Load the image from buffer
  try {
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
    const imgResult = canvas.toBuffer();
    // const frameFileName = `imgs/frame_${currentTime}.jpg`;
    // fs.writeFileSync(frameFileName, imageBufferResult);

    callback?.(imgResult);
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

  //   await handleLoadModels();

  const videoUrl = "https://cdn.shinobi.video/videos/theif4.mp4";
  // const videoUrl =
  //   "http://192.168.100.252:8989/get-stream/cWEtc2l0ZQ--/cnRzcDovL3JhcHRvcjpSYXB0b3IxMjMhQDE5Mi4xNjguMTAwLjEzMjo1NTQvY2FtL3JlYWxtb25pdG9yP2NoYW5uZWw9MSZzdWJ0eXBlPTAmdW5pY2FzdD10cnVlJnByb3RvPU9udmlm";

  const rtspStreamUrl =
    "rtsp://raptor:Raptor123!@192.168.100.181:554/cam/realmonitor?channel=1&subtype=0&unicast=true&proto=Onvif";

  const imagesStream = new Readable({
    read(size) {},
  });
  const queue = async.queue(worker, 1);
  processFramesFromRTSPStream(videoUrl, queue, imagesStream, res);

  pipeline(imagesStream, res, (err) => {
    if (err) {
      console.error("Error piping ffmpeg output to response:", err);
      res.end();
    }
  });
});
