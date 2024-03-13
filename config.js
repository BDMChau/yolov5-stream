const config = {
  plug: "RaptorAI",
  host: "localhost",
  port: 8080,
  key: "1979d4a2e2fa094e5e06c7e5f78eda0c51937339d0e22b8e7da758ca8fff",
  gpu: true,
  mode: "client",
  type: "detector",
  debugLog: false,
  enabled: false,
  redFlagAttire: ["Hat", "Sunglasses"],
  redFlagProducts: ["Bottle", "Tin can"],
  redFlagContainers: ["Suitcase", "Backpack", "Handbag"],
  tfjsBuild: "gpu",

  cameraType: 1, // 0: AXIS, 1: AMCREST
  IP: "192.168.100.111",
  WEIGHT_FILE_NAME: "yolov5s.pt",
};

module.exports = config;
