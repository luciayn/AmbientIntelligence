const appSection = document.querySelector("#app");

const webcamEl = document.querySelector("#webcam");
const enableWebcamButton = document.querySelector("#webcamButton");
const btnTrain = document.querySelector("#train");
const btnPredict = document.querySelector("#predict");
const btnsDataCollector = document.querySelectorAll("button.dataCollector");

const statusEl = document.querySelector("#status");

const MOBILENET_MODEL_URL =
  "https://www.kaggle.com/models/google/mobilenet-v3/TfJs/large-100-224-feature-vector/1";
  //https://tfhub.dev/google/tfjs-model/imagenet/mobilenet_v3_large_100_224/feature_vector/5/default/1

const MOBILENET_INPUT_HEIGHT = 224;
const MOBILENET_INPUT_WIDTH = 224;

const MOBILENET_NUM_FEATURES = 1280;

const CLASS_NAMES = [];
const NUM_CLASSES = 2;

let trainingDataInputs = [];
let trainingDataOutputs = [];

let model;
let webcam;
let classifier;

let videoPlaying = false;

const STOP_DATA_GATHER = -1;
let gatherDataState = STOP_DATA_GATHER;

const EXAMPLES_COUNT = new Array(NUM_CLASSES);



function initTFJS() {
  if (typeof tf === "undefined") {
    throw new Error("TensorFlow.js not loaded");
  }
}

function getUserMediaSupported() {
  return Boolean(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

async function enableCam(event) {
  if (!model) return;

  event.target.classList.add("removed");
  webcam = await tf.data.webcam(webcamEl, {});
  videoPlaying = true;
}

function createClassifier() {
  let model = tf.sequential();
  model.add(
    tf.layers.dense({
      inputShape: [MOBILENET_NUM_FEATURES],
      units: 128,
      activation: "relu"
    })
  );
  model.add(tf.layers.dense({ units: NUM_CLASSES, activation: "softmax" }));

  model.summary();

  model.compile({
    optimizer: "adam",
    loss: NUM_CLASSES === 2 ? "binaryCrossentropy" : "categoricalCrossentropy",
    metrics: ["accuracy"]
  });

  console.log("Classifier created successfully");

  return model;
}

function logProgress(epoch, logs) {
  console.log(`Epoc ${epoch}`, logs);
}

async function trainClassifier() {
  if (!classifier) return;
  btnsDataCollector.forEach((bdc) => {
    bdc.disabled = true;
  });

  tf.util.shuffleCombo(trainingDataInputs, trainingDataOutputs);
  let outputsAsTensor = tf.tensor1d(trainingDataOutputs, "int32");
  let oneHotOutputs = tf.oneHot(outputsAsTensor, CLASS_NAMES.length);
  let inputsAsTensor = tf.stack(trainingDataInputs);
  inputsAsTensor.print(true);

  await classifier.fit(inputsAsTensor, oneHotOutputs, {
    shuffle: true,
    batchSize: 5,
    epochs: 10,
    callbacks: { onEpochEnd: logProgress }
  });

  outputsAsTensor.dispose();
  oneHotOutputs.dispose();
  inputsAsTensor.dispose();

  btnTrain.disabled = true;
  btnPredict.disabled = false;
  console.log("Training completed");

}


async function loadMobileNetModel() {
  //const net = await mobilenet.load();
  const net = await tf.loadGraphModel(MOBILENET_MODEL_URL, { fromTFHub: true });
  console.log("MobileNet loaded successfully");
  appSection.classList.toggle("removed");
  return net;
}

function preprocess(imageTensor) {
  return tf.tidy(() => {
    const widthToHeight = imageTensor.shape[1] / imageTensor.shape[0];
    let squareCrop;
    if (widthToHeight > 1) {
      const heightToWidth = imageTensor.shape[0] / imageTensor.shape[1];
      const cropTop = (1 - heightToWidth) / 2;
      const cropBottom = 1 - cropTop;
      squareCrop = [[cropTop, 0, cropBottom, 1]];
    } else {
      const cropLeft = (1 - widthToHeight) / 2;
      const cropRight = 1 - cropLeft;
      squareCrop = [[0, cropLeft, 1, cropRight]];
    }
    const crop = tf.image.cropAndResize(
      tf.expandDims(imageTensor),
      squareCrop,
      [0],
      [224, 224]
    );

    return crop.div(255);
  });
}

function postprocess(result) {
  tf.tidy(() => {
    const classNumber = result.argMax().dataSync()[0];
    const confidence = result.max().dataSync()[0];
    statusEl.innerText = `Clasificación: ${
      CLASS_NAMES[classNumber]
    } con ${Math.floor(confidence * 100)}% de confianza`;
  });
}

async function predict() {
  while (true) {
    const videoFrameAsTensor = await webcam.capture();
    //const features = model.infer(imageFrameAsTensor, true);
    const preprocessedInputTensor = preprocess(videoFrameAsTensor);
    const featuresTensor = model.predict(preprocessedInputTensor);
    featuresTensor.print(true);
    //const result = await classifier.predictClass(features);
    const outputTensor = await classifier.predict(featuresTensor);

    postprocess(outputTensor.squeeze());

    videoFrameAsTensor.dispose();
    preprocessedInputTensor.dispose();
    featuresTensor.dispose();
    outputTensor.dispose();
    await tf.nextFrame();
  }
}

function gatherDataForClass() {
  let classNumber = parseInt(this.getAttribute("data-1hot"));
  gatherDataState =
    gatherDataState === STOP_DATA_GATHER ? classNumber : STOP_DATA_GATHER;
  dataGatherLoop();
}

async function dataGatherLoop() {
  if (videoPlaying && gatherDataState !== STOP_DATA_GATHER) {
    const videoFrameAsTensor = await webcam.capture();
    const preprocessedInputTensor = preprocess(videoFrameAsTensor);
    const features = model.predict(preprocessedInputTensor);
    features.print(true);
    //classifier.addExample(features, gatherDataState);
    trainingDataInputs.push(features.squeeze());
    trainingDataOutputs.push(gatherDataState);

    videoFrameAsTensor.dispose();
    preprocessedInputTensor.dispose();
    features.dispose();

    if (EXAMPLES_COUNT[gatherDataState] === undefined) {
      EXAMPLES_COUNT[gatherDataState] = 0;
    }
    EXAMPLES_COUNT[gatherDataState]++;

    statusEl.innerText = "";
    for (let n = 0; n < NUM_CLASSES; n++) {
      statusEl.innerText += `Ejemplos para ${CLASS_NAMES[n]} : ${EXAMPLES_COUNT[n]}.\n`;
    }

    await tf.nextFrame();
    dataGatherLoop();
  }
}


async function app() {
  if (!getUserMediaSupported()) {
    console.warn("getUserMedia() is not supported by your browser");
    return; // Exit if webcam is not supported
  }

  model = await loadMobileNetModel();
  classifier = createClassifier();

  enableWebcamButton.addEventListener("click", enableCam);
  btnPredict.addEventListener("click", predict);
  btnTrain.addEventListener("click", trainClassifier);
  btnsDataCollector.forEach((bdc) => {
    bdc.addEventListener("mousedown", gatherDataForClass);
    bdc.addEventListener("mouseup", gatherDataForClass);
    CLASS_NAMES.push(bdc.getAttribute("data-name"));
  });
}

(async function initApp() {
  try {
    initTFJS();
    await app();

  } catch (error) {
    console.error(error);
  }

}());
