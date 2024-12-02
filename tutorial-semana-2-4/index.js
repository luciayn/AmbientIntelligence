const webcamEl = document.querySelector("#webcam");
const canvas = document.querySelector("#canvas");
const outputMessageEl = document.querySelector("#outputMessage");

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

        // Reconocer gesto de "pulgar arriba"
        const isThumbUp = detectThumbUpGesture(landmarks);
        if (isThumbUp) {
          outputMessageEl.innerText = "Gesto: Pulgar Arriba";
        } else {
          outputMessageEl.innerText = "Gesto: No Detectado";
        }

        // Calcular y mostrar la distancia entre la punta del pulgar y la punta del dedo índice
        const distance = calculateDistance(landmarks[4], landmarks[8]); // Pulgar y Dedo índice
        displayDistance(context, distance); // Mostrar la distancia en el lienzo
      });
    }

    img.dispose();
    await tf.nextFrame();
  }
}

function detectThumbUpGesture(landmarks) {
  // Coordenadas relevantes del pulgar
  const thumbBase = landmarks[0];   // Punto 0 (base del pulgar)
  const thumbTip = landmarks[4];    // Punto 4 (punta del pulgar)

  // Condición para que el pulgar esté arriba:
  // El pulgar debe estar por encima de su base (Punto 0) en el eje Y.
  const thumbUp = thumbTip.y < thumbBase.y && thumbTip.y < landmarks[1].y; // El pulgar debe estar más arriba que el punto 1.

  // También puedes comprobar que los otros dedos estén doblados o hacia abajo, si quieres ser más estricto:
  const otherFingersDown = landmarks[8].y > landmarks[5].y && landmarks[12].y > landmarks[9].y &&
                           landmarks[16].y > landmarks[13].y && landmarks[20].y > landmarks[17].y; 

  return thumbUp && otherFingersDown; // Si el pulgar está arriba y los otros dedos están abajo, es pulgar arriba.
}

// Función para calcular la distancia entre dos puntos
function calculateDistance(point1, point2) {
  const dx = point2.x - point1.x;
  const dy = point2.y - point1.y;
  return Math.sqrt(dx * dx + dy * dy);
}

// Función para mostrar la distancia en el lienzo
function displayDistance(context, distance) {
  context.font = '20px Arial';
  context.fillStyle = 'white';
  context.fillText(`Distancia: ${distance.toFixed(2)} px`, 10, 30);
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
  const connections_dict = {
    'thumb': [[0, 1], [1, 2], [2, 3], [3, 4]], 
    'index': [[0, 5], [5, 6], [6, 7], [7, 8]],
    'middle': [[0, 9], [9, 10], [10, 11], [11, 12]],
    'ring': [[0, 13], [13, 14], [14, 15], [15, 16]],
    'pinky': [[0, 17], [17, 18], [18, 19], [19, 20]],
  };
  
  const fingerColors = {
    'thumb': 'blue',
    'index': 'green',
    'middle': 'yellow',
    'ring': 'purple',
    'pinky': 'pink'
  }
  // Loop through each finger and draw lines
  Object.keys(connections_dict).forEach((finger) => {
    const color = fingerColors[finger];
    context.strokeStyle = color;

    // Draw the connections (lines between landmarks)
    connections_dict[finger].forEach(([start, end]) => {
      const startPoint = landmarks[start];
      const endPoint = landmarks[end];

      context.beginPath();
      context.moveTo(startPoint.x, startPoint.y); // Move to the start landmark
      context.lineTo(endPoint.x, endPoint.y);    // Draw to the end landmark
      context.lineWidth = 2;
      context.stroke();
    });
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
      // outputMessageEl.innerText = error.message;
    }
  }

}());
