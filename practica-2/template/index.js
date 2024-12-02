const webcamEl = document.querySelector("#webcam");
const canvas = document.querySelector("#canvas");
const outputMessageEl = document.querySelector("#outputMessage");
let videoElement = document.querySelector('video');3
let previousWristPosition = { x: 0, y: 0, z: 0 };
let palmOrientation = { x: 0, y: 0, z: 0 };
function initTFJS() {
  if (typeof tf === "undefined") {
    throw new Error("TensorFlow.js not loaded");
  }
}

async function app() {
  const model = handPoseDetection.SupportedModels.MediaPipeHands;
  const detectorConfig = {
    runtime: 'mediapipe',
    modelType: 'full',
    maxHands: 2,
    solutionPath: 'https://cdn.jsdelivr.net/npm/@mediapipe/hands'
    // or 'base/node_modules/@mediapipe/hands' in npm.
  };
  detector = await handPoseDetection.createDetector(model, detectorConfig);


  const webcam = await tf.data.webcam(webcamEl, {
    resizeWidth: 252,
    resizeHeight: 252
  });

  const camerabbox = webcamEl.getBoundingClientRect();
  canvas.style.top = camerabbox.y + "px";
  canvas.style.left = camerabbox.x + "px";

  const context = canvas.getContext("2d");

  while (true) {

    const img = await webcam.capture();
    // Prediction
    const hands = await detector.estimateHands(img, { flipHorizontal: true });
    console.log(hands); 

    context.clearRect(0, 0, canvas.width, canvas.height);
    if (hands.length > 0) {
      hands.forEach(hand => {
        const landmarks = hand.keypoints;
        drawHandLandmarks(context, landmarks);
      
        // Detect palm orientation and movement
        const wrist = landmarks[0];
        const palmOrientation = calculatePalmOrientation(landmarks);
        // outputMessageEl.innerText = wrist.x;
        const distanceMoved = detectHorizontalMovement(wrist);

        // Control video playback based on palm orientation and wrist movement
        if (palmOrientation.x > 0.9) {  // Palm oriented to the left
          // Adjust video based on wrist movement
          if (distanceMoved > 0) {  // Only move if the wrist has moved significantly
            videoElement.currentTime += distanceMoved * 10;  // Scale wrist movement
          }
          else {
            videoElement.currentTime -= distanceMoved * 10;  // Scale wrist movement
          }
        }
      });
    }

    img.dispose();
    await tf.nextFrame();
  }
}

function calculatePalmOrientation(landmarks) {
  const wrist = landmarks[0]; // Wrist
  const indexMCP = landmarks[5]; // Index MCP
  const pinkyMCP = landmarks[17]; // Pinky MCP

  const vector1 = { x: indexMCP.x - wrist.x, y: indexMCP.y - wrist.y }; // Vector from wrist to index
  const vector2 = { x: pinkyMCP.x - wrist.x, y: pinkyMCP.y - wrist.y }; // Vector from wrist to pinky
  const crossProduct = vector1.x * vector2.y - vector1.y * vector2.x;
  
  // Calculate the cross product to get the orientation
  const orientation = crossProduct > 0 ? "left" : "right";
  outputMessageEl.innerText = orientation;
  return orientation;
}

// Detect wrist movement in the horizontal direction
function detectHorizontalMovement(wrist) {
  const distanceMoved = wrist.x - previousWristPosition.x;
  previousWristPosition = wrist;

  return distanceMoved;
}


function drawHandLandmarks(context, landmarks) {
  const color = 'red';
  // Define the connections between landmarks (bones of the hand)
  const connections = [
    [0, 1], [1, 2], [2, 3], [3, 4],       // Thumb
    [0, 5], [5, 6], [6, 7], [7, 8],       // Index Finger
    [0, 9], [9, 10], [10, 11], [11, 12],  // Middle Finger
    [0, 13], [13, 14], [14, 15], [15, 16], // Ring Finger
    [0, 17], [17, 18], [18, 19], [19, 20]  // Pinky Finger
  ];

  // Draw lines for each connection
  connections.forEach(([start, end]) => {
    const startPoint = landmarks[start];
    const endPoint = landmarks[end];

    context.beginPath();
    context.moveTo(startPoint.x, startPoint.y); // Move to the start landmark
    context.lineTo(endPoint.x, endPoint.y);    // Draw to the end landmark
    context.lineWidth = 2;
    context.strokeStyle = 'blue';
    context.stroke();
  });

  // Draw circles for each landmark
  landmarks.forEach((landmark) => {
    drawCircle(context, landmark.x, landmark.y, 3, color);
  });
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
