const MODEL_URL =
  "https://tfhub.dev/google/tfjs-model/movenet/singlepose/lightning/4";
const webcamEl = document.querySelector("#webcam");
const canvas = document.querySelector("#canvas");


function initTFJS() {
  if (typeof tf === "undefined") {
    throw new Error("TensorFlow.js not loaded");
  }
}

async function app() {
  const model = await tf.loadGraphModel(MODEL_URL, { fromTFHub: true });
  const webcam = await tf.data.webcam(webcamEl, {
    resizeWidth: 192,
    resizeHeight: 192
  });

  const camerabbox = webcamEl.getBoundingClientRect();
  canvas.style.top = camerabbox.y + "px";
  canvas.style.left = camerabbox.x + "px";

  const context = canvas.getContext("2d");

  /* The camera is not mirrored,
   * you need to mirror the canvas to make it look normal (x axis)
   */
  context.translate(webcamEl.width, 0);
  context.scale(-1, 1);

  while (true) {
   
    const img = await webcam.capture();

    // Prediction
    const prediction = await model.predict(img.toInt().expandDims());
    const arrayOut = await prediction.array();
    const points = arrayOut[0][0];

    context.clearRect(0, 0, canvas.width, canvas.height);
    for (let i = 0; i < points.length; i++) {
      const point = points[i];

      if (point[2] > 0.4) {
        drawCircle(context, point[1] * 252, point[0] * 252, 5, "#003300");
      }
    }

    img.dispose();
    await tf.nextFrame();
  }
}

function drawCircle(context, cx, cy, radius, color) {
  context.beginPath();
  context.arc(cx, cy, radius, 0, 2 * Math.PI, false);
  context.fillStyle = "red";
  context.fill();
  context.lineWidth = 1;
  context.strokeStyle = color;
  context.stroke();
}

(async function initApp() {
  try {
    initTFJS();
    await app();
  } catch (error) {
    console.error(error);
    if (outputMessageEl) {
      outputMessageEl.innerText = error.message;
    }
  }

}());
