const MODEL_URL =
  "https://tfhub.dev/google/tfjs-model/movenet/singlepose/thunder/4";
const webcamEl = document.querySelector("#webcam");
const canvas = document.querySelector("#canvas");
const imgEl = document.querySelector("#img");


let WIDTH = 256;
let HEIGHT = 256;



function initTFJS() {
  if (typeof tf === "undefined") {
    throw new Error("TensorFlow.js not loaded");
  }
}

async function app() {
  const model = await tf.loadGraphModel(MODEL_URL, { fromTFHub: true });
  const webcam = await tf.data.webcam(webcamEl, {
    resizeWidth: 256,
    resizeHeight: 256
  });

  const camerabbox = webcamEl.getBoundingClientRect();
  canvas.style.top = camerabbox.y + "px";
  canvas.style.left = camerabbox.x + "px";

  const context = canvas.getContext("2d");
  canvas.width = webcamEl.videoWidth;
  canvas.height = webcamEl.videoHeight;
  WIDTH = webcamEl.videoWidth;
  HEIGHT = webcamEl.videoHeight;
  console.log(WIDTH, HEIGHT);
  /* The camera is not mirrored,
   * you need to mirror the canvas to make it look normal (x axis)
   */
  context.translate(WIDTH, 0);
  context.scale(-1, 1);

  while (true) {
  // if (imgEl.complete) {

    const img = await webcam.capture();
    //const img = await tf.browser.fromPixels(imgEl);

    // Prediction
    const prediction = await model.predict(img.resizeBilinear([256, 256]).toInt().expandDims());
    const arrayOut = await prediction.array();
    const points = arrayOut[0][0];
    context.clearRect(0, 0, WIDTH, HEIGHT);
    drawKeypoints(points, context);
    const isPrayPoseValid = isPrayPose(points);
    if(isPrayPoseValid){
      document.body.style.backgroundColor = "green";
    } else {
      document.body.style.backgroundColor = "white";
    }
    //drawElbowAngles(points, context);

    img.dispose();
    prediction.dispose();
  

  // Set the canvas size to match the video size
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

function drawKeypoints(points, context) {
  // Define connections between keypoints for drawing skeleton lines
  const CONNECTIONS = [
    [0, 1], [0, 2], [2, 4], [1, 3],         // nose to eyes and ears
    [5, 6],                                 // shoulders
    [5, 7], [7, 9],                         // left arm
    [6, 8], [8, 10],                        // right arm
    [5, 11], [6, 12],                       // shoulders to hips
    [11, 12],                               // hips
    [11, 13], [13, 15],                     // left leg
    [12, 14], [14, 16]                      // right leg
  ];

  // Draw lines connecting keypoints
  CONNECTIONS.forEach(([start, end]) => {
    const [y1, x1, confidence1] = points[start];
    const [y2, x2, confidence2] = points[end];

    if (confidence1 > 0.2 && confidence2 > 0.2) {
      context.beginPath();
      context.moveTo(x1 * WIDTH, y1 * HEIGHT);
      context.lineTo(x2 * WIDTH, y2 * HEIGHT);
      context.lineWidth = 2;
      context.strokeStyle = "purple";
      context.stroke();
    }
  });

  // Draw all keypoints as circles
  points.forEach(([y, x, confidence], index) => {
    if (confidence > 0.2) {
      drawCircle(context, x * WIDTH, y * HEIGHT, 5, "orange");
    }
  });
}

// Circle-drawing helper function
function drawCircle(context, cx, cy, radius, color) {
  context.beginPath();
  context.arc(cx, cy, radius, 0, 2 * Math.PI, false);
  context.fillStyle = color;
  context.fill();
  context.lineWidth = 1;
  context.strokeStyle = color;
  context.stroke();
}

function calculateAngle(A, C, B) {
  const AC = [C[0] - A[0], C[1] - A[1]];
  const BC = [C[0] - B[0], C[1] - B[1]];

  // Calcular el producto punto y la magnitud de los vectores
  const dotProduct = AC[0] * BC[0] + AC[1] * BC[1];

  const magAC = Math.sqrt(AC[0] ** 2 + AC[1] ** 2);
  const magBC = Math.sqrt(BC[0] ** 2 + BC[1] ** 2);
  
  // Calcular el coseno del ángulo
  let cosTheta = dotProduct / (magAC * magBC);
  
  // Calcular el ángulo en radianes y convertirlo a grados
  const angle = Math.acos(cosTheta) * (180 / Math.PI);
  return angle;
}


function isPrayPose(points) {
  const convertedPoints = points.map(([y, x]) => [x, y]);
  const LEFT_SHOULDER = convertedPoints[5];
  const RIGHT_SHOULDER = convertedPoints[6];
  const LEFT_ELBOW = convertedPoints[7];
  const RIGHT_ELBOW = convertedPoints[8];
  const LEFT_HAND = convertedPoints[9];
  const RIGHT_HAND = convertedPoints[10];

  const areHandsTogether = Math.abs(LEFT_HAND[0] - RIGHT_HAND[0]) < 0.1;
  const leftElbowAngle = calculateAngle(LEFT_SHOULDER, LEFT_ELBOW, LEFT_HAND);
  const rightElbowAngle = calculateAngle(RIGHT_SHOULDER, RIGHT_ELBOW, RIGHT_HAND);
  const isLeftElbowBent = leftElbowAngle > 45 && leftElbowAngle < 70;
  const isRightElbowBent = rightElbowAngle > 45 && rightElbowAngle < 70;

  console.log(areHandsTogether, isLeftElbowBent, isRightElbowBent);
  return areHandsTogether && isLeftElbowBent && isRightElbowBent;
}

function isTreePose(points) {
  const convertedPoints = points.map(([y, x]) => [x, y]);
  const LEFT_HIP = convertedPoints[11];
  const RIGHT_HIP = convertedPoints[12];
  const LEFT_KNEE = convertedPoints[13];
  const RIGHT_KNEE = convertedPoints[14];
  const LEFT_ANKLE = convertedPoints[15];
  const RIGHT_ANKLE = convertedPoints[16];
  const LEFT_SHOULDER = convertedPoints[5];
  const RIGHT_SHOULDER = convertedPoints[6];
  const LEFT_HAND = convertedPoints[9];
  const RIGHT_HAND = convertedPoints[10];
  const HEAD = convertedPoints[0];

  // Regla: Pierna de Apoyo 
  const rightLegAngle = calculateAngle(RIGHT_HIP, RIGHT_KNEE, RIGHT_ANKLE);
  const isRightLegStraight = rightLegAngle > 170;
  const isRightAnkleAligned = Math.abs(RIGHT_ANKLE[0] - RIGHT_HIP[0]) < 0.1;

  // Regla: Pierna Doblada
  const leftLegAngle = calculateAngle(LEFT_HIP, LEFT_KNEE, LEFT_ANKLE);
  const isLeftKneeBentOutward = leftLegAngle > 20 && leftLegAngle < 90; 

  const isXAligned = Math.abs(LEFT_ANKLE[0] - RIGHT_KNEE[0]) < 0.1;
  const isAboveKnee = LEFT_ANKLE[1] < (RIGHT_KNEE[1] - 0.1); 
  const isBelowKnee = LEFT_ANKLE[1] > (RIGHT_KNEE[1] + 0.1); 
  const isLeftFootPositioned = isXAligned && (isAboveKnee || isBelowKnee);

  // Regla: Columna Recta
  const shoulderMidpointX = (LEFT_SHOULDER[0] + RIGHT_SHOULDER[0]) / 2;
  const hipMidpointX = (LEFT_HIP[0] + RIGHT_HIP[0]) / 2;
  const isTorsoAligned = Math.abs(shoulderMidpointX - hipMidpointX) < 0.05;
 
  // Regla: Hombros al Mismo Nivel
  const areShouldersLevel = Math.abs(LEFT_SHOULDER[1] - RIGHT_SHOULDER[1]) < 0.05;

  // Regla: Manos Levantadas Sobre la Cabeza
  const handsAboveHead = LEFT_HAND[1] < HEAD[1] && RIGHT_HAND[1] < HEAD[1];
  const areHandsTogether = Math.abs(LEFT_HAND[0] - RIGHT_HAND[0]) < 0.1 &&
    Math.abs(LEFT_HAND[1] - RIGHT_HAND[1]) < 0.1;

  // Regla: Cabeza Erguida
  const isHeadUpright = Math.abs(HEAD[0] - ((LEFT_SHOULDER[0] + RIGHT_SHOULDER[0]) / 2)) < 0.05;

  // Evaluación Final de la Postura del Árbol
  const isTreePoseValid = isRightLegStraight && isRightAnkleAligned &&
    isLeftKneeBentOutward && isLeftFootPositioned &&
    isTorsoAligned && areShouldersLevel &&
    handsAboveHead && areHandsTogether &&
    isHeadUpright;

  return isTreePoseValid;
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
