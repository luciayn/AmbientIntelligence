const webcamEl = document.querySelector("#webcam");
const canvas = document.querySelector("#canvas");
const outputMessageEl = document.querySelector("#outputMessage");
let videoElement = document.querySelector("#mp4");
let previousWristPosition = { x: 0, y: 0, z: 0 };
let frameCounter = 0;
let previousWristY = 0;

function initTFJS() {
  if (typeof tf === "undefined") {
    throw new Error("TensorFlow.js not loaded");
  }
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

///////////////////////////////////////////////
// GESTO 1: Mover el vídeo adelante o atrás //
//////////////////////////////////////////////

// Calcular el vector normal para conocer la orientación de la palma (se han seguido las instrucciones del lab3)
function calcularVectorNormal(landmarks3D) { 
  // Paso 1: Normalizar los puntos y calcular la magnitud máxima
  const magnitudes = landmarks3D.map(point => 
      Math.sqrt(point.x ** 2 + point.y ** 2 + point.z ** 2)
  );
  const maxMagnitud = Math.max(...magnitudes);
  // Paso 2: Escalar los puntos para valores entre 0 y 1
  const puntosNormalizados = landmarks3D.map(point => [
      point.x /= maxMagnitud,
      point.y /= maxMagnitud,
      point.z /= maxMagnitud,
  ]);
  // Paso 3: Seleccionar los puntos clave
  const wrist = puntosNormalizados[0]; // Muñeca
  const indexMCP = puntosNormalizados[5]; // Base del índice
  const pinkyMCP = puntosNormalizados[17]; // Base del meñique
  // Paso 4: Construir los dos vectores que definen el plano de la palma
  const vector1 = [
      indexMCP[0] - wrist[0],
      indexMCP[1] - wrist[1],
      indexMCP[2] - wrist[2],
  ];
  const vector2 = [
      pinkyMCP[0] - wrist[0],
      pinkyMCP[1] - wrist[1],
      pinkyMCP[2] - wrist[2],
  ];
  // Paso 5: Calcular el vector normal como el producto vectorial
  const vectorNormal = [
      vector1[1] * vector2[2] - vector1[2] * vector2[1],
      vector1[2] * vector2[0] - vector1[0] * vector2[2],
      vector1[0] * vector2[1] - vector1[1] * vector2[0],
  ];
  return vectorNormal;
}

// Detectar el movimiento horizontal de la muñeca
function detectHorizontalMovement(wrist) {
  const distanceMoved = wrist.x - previousWristPosition.x;
  previousWristPosition = wrist;  // Actualizar posición anterior de la muñeca
  return distanceMoved;
}

// Detectar palma abierta con dedos apuntando hacia arriba
function areFingersExtendedUpwards(landmarks) {
  const extendedFingers = [8, 12, 16, 20].every((tipIndex, i) => {
      const baseIndex = tipIndex - 2; // Articulación base del dedo
      // Si la punta del dedo se encuentra por encima de la base, este estará apuntando hacia arriba
      return landmarks[tipIndex].y<landmarks[baseIndex].y;
  });
  return extendedFingers;
}


//////////////////////////////
// GESTO 2: Cerrar el vídeo //
//////////////////////////////

// Comprobar si la mano está cerrada en puño
function isFist(landmarks) {
  const fingerTips = [4, 8, 12, 16, 20];  // Puntas de los dedos
  const fingerBases = [3, 6, 10, 14, 18]; // Bases de los dedos
  
  // Calcular las distancias entre las puntas y bases de cada dedo
  const distances = fingerTips.map((tipIndex, i) => {
      const baseIndex = fingerBases[i];
      const distance = Math.sqrt(
        Math.pow(landmarks[tipIndex].x - landmarks[baseIndex].x, 2) +
        Math.pow(landmarks[tipIndex].y - landmarks[baseIndex].y, 2) +
        Math.pow(landmarks[tipIndex].z - landmarks[baseIndex].z, 2)
    );
      return distance;
  });
  // console.log(`Distances: ${distances}`);

  // Promedio de las distancias
  const averageDistance = distances.reduce((sum, d) => sum + d, 0) / distances.length;
  // outputMessageEl.innerText = `averageDistance: ${averageDistance}`;

  // Verificar si el promedio está por debajo de un umbral
  const fistThreshold = 0.38;
  return averageDistance < fistThreshold;
}


///////////////////////////////////////
// GESTO 3: Subir o bajar el volumen //
///////////////////////////////////////

// Comprobar la orientación de la palma izquierda hacia la cámara
function isLeftHandFacingCamera(landmarks3D) {
  const normal = calcularVectorNormal(landmarks3D);
  return normal[2] > 0; // Positivo: palma orientada hacia el usuario, Negativo: palma orientada hacia la cámara
}

// Comprobar si la mano derecha apunta hacia la izquierda
function isRightHandPointingLeft(landmarks3D) {
  const indexTip = landmarks3D[8];
  const indexBase = landmarks3D[6];
  const wrist = landmarks3D[0];
  // console.log(`wrist.x: ${wrist.x}, indexTip.x: ${indexTip.x}, indexTip.x < indexBase.x: ${indexTip.x < indexBase.x}, Math.abs(indexTip.x - wrist.x): ${Math.abs(indexTip.x - wrist.x)>1}`);
  
  // Comprobar si el índice está extendido
  const isIndexExtended = indexTip.x < indexBase.x && Math.abs(indexTip.x - wrist.x) > 1;
  // Comprobar si el resto de dedos están doblados
  const otherFingers = [12, 16, 20]; // Medio, anular y meñique
  const areOtherFingersBent = otherFingers.every(tipIndex => {
    const baseIndex = tipIndex - 2; // Base del dedo
    return landmarks3D[tipIndex].x > landmarks3D[baseIndex].x; // Si la punta se sitúa a la derecha de la base, el dedo está doblado
  });
  // Comprobar que el pulgar está doblado
  const thumbTip = landmarks3D[4];
  const thumbBase = landmarks3D[3];
  const isThumbBent = thumbTip.y > thumbBase.y;
  // console.log(`isIndexExtended: ${isIndexExtended}, areOtherFingersBent: ${areOtherFingersBent}, isThumbBent: ${isThumbBent}`);
  
  return isIndexExtended && areOtherFingersBent && isThumbBent;
}

// Ajustar el volumen acorde al movimiento de la muñeca hacia arriba o abajo
function adjustVolume(leftHand, rightHand, previousWristY) {
  const LeftHandFacingCamera = isLeftHandFacingCamera(leftHand.keypoints3D);
  const LeftHandExtended = areFingersExtendedUpwards(leftHand.keypoints3D);
  const RightHandPointingLeft = isRightHandPointingLeft(rightHand.keypoints3D);
  const wristY = rightHand.keypoints[0].y;
  console.log(`wristY: ${wristY}, previousWristY: ${previousWristY}`);

  // Ajustar el volumen en caso de estar la mano izquierda extendida y orientada hacia la cámara, y la mano derecha apuntando hacia la izquierda
  if (LeftHandFacingCamera && LeftHandExtended && RightHandPointingLeft) {
    console.log(`LeftHandExtended: ${LeftHandExtended}, RightHandPointingLeft: ${RightHandPointingLeft}`);
    // Calcular la distancia de movimiento de la muñeca hacia arriba o abajo
    const distanceMoved = wristY - previousWristY;
    console.log(`distanceMoved: ${distanceMoved}`);

    if (distanceMoved < 0) { // Movimiento hacia arriba: subir el volumen
      videoElement.volume = Math.min(videoElement.volume + 0.1, 1);
      outputMessageEl.innerText = "Volumen: Subiendo";
    } else if (distanceMoved > 0) { // Movimiento hacia abajo: bajar el volumen
      videoElement.volume = Math.max(videoElement.volume - 0.1, 0);
      outputMessageEl.innerText = "Volumen: Bajando";
    }
  }
  return wristY;
}

/////////////////////////////////////////////////
// GESTO 4: Reproducir el vídeo a velocidad x2 //
/////////////////////////////////////////////////

// Detectar el símbolo de paz (dedos índice y medio extendidos, y dedos pulgar, anular y meñique doblados)
function isPeaceSign(landmarks) {
  // Índices de las puntas de los dedos
  const thumbTip = landmarks[4];
  const indexTip = landmarks[8];
  const middleTip = landmarks[12];
  const ringTip = landmarks[16];
  const pinkyTip = landmarks[20];

  // Índices de las bases de los dedos
  const thumbBase = landmarks[3];
  const indexBase = landmarks[5];
  const middleBase = landmarks[9];
  const ringBase = landmarks[13];
  const pinkyBase = landmarks[17];

  // Comprobar si los dedos índice y medio están extendidos
  const isIndexExtended = indexTip.y < indexBase.y;
  const isMiddleExtended = middleTip.y < middleBase.y;

  // Comprobar si los dedos pulgar, anular y meñique están doblados
  const isThumbBent = thumbTip.x > thumbBase.x;
  const isRingBent = ringTip.y > ringBase.y;
  const isPinkyBent = pinkyTip.y > pinkyBase.y;

  return isIndexExtended && isMiddleExtended && isThumbBent && isRingBent && isPinkyBent;
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
    
    // Procesar sólamente 1 de cada 5 frames: Suavizar los movimientos detectados y evitar detecciones erráticas
    if (frameCounter%5 === 0) {
      // Prediction
      const hands = await detector.estimateHands(img, { flipHorizontal: true });
      console.log(hands); 

      context.clearRect(0, 0, canvas.width, canvas.height);
      if (hands.length > 0) {
        hands.forEach(hand => {
          const landmarks2D = hand.keypoints;
          const landmarks3D = hand.keypoints3D;
          drawHandLandmarks(context, landmarks2D);
          
          // GESTO 1: Mover el vídeo adelante o atrás
          const normal = calcularVectorNormal(landmarks3D); // Calcular el vector normal
          // console.log(`vectorNormal: ${normal[0]}`);

          const FingersExtendedUpwards = areFingersExtendedUpwards(landmarks3D); // Comprobar si los dedos están extendidos hacia arriba
          const wrist = landmarks2D[0]; // Componente x
          const distanceMoved = detectHorizontalMovement(wrist);
          // console.log(`FingersExtendedUpwards: ${FingersExtendedUpwards}`);
          // console.log(`Normal[0]: ${normal[0]}, distanceMoved: ${distanceMoved}`);

          if (FingersExtendedUpwards && hand.handedness=='Right') {  // Palma derecha abierta con los dedos apuntando hacia arriba
            if (normal[0] > 0.4) {  // Palma orientada hacia la izquierda, componente x mayor que 0.4
              if (distanceMoved > 2) {
                videoElement.currentTime += 3;  // Avanzar 3 segundos
                outputMessageEl.innerText = "Avanzando vídeo";
              }
              else if (distanceMoved < -2) {
                console.log(`distanceMoved: ${distanceMoved}`);
                videoElement.currentTime -= 3;  // Retroceder 3 segundos
                outputMessageEl.innerText = "Retrocediendo vídeo";
              }
            }
          }

          // GESTO 2: Cerrar el vídeo
          const isMakingFist = isFist(landmarks3D);
          if (isMakingFist && hand.handedness=='Right'){
            // Detener y ocultar el reproductor
            videoElement.pause();  // Pausar el vídeo
            videoElement.style.display = "none";  // Ocultar el vídeo
            outputMessageEl.innerText = "Vídeo Pausado y Ocultado";
          }
          
          // GESTO 3: Subir o bajar el volumen
          const leftHand = hands.find(h => h.handedness === 'Left');
          const rightHand = hands.find(h => h.handedness === 'Right');
          if (leftHand && rightHand) {
            previousWristY = adjustVolume(leftHand, rightHand, previousWristY);
          }

          // GESTO 4: Reproducir el vídeo a velocidad x2
          if (rightHand && isPeaceSign(rightHand.keypoints)) {
            videoElement.playbackRate = 2.0; // Duplica la velocidad
            outputMessageEl.innerText = "Velocidad: x2";
          } else {
            videoElement.playbackRate = 1.0; // Vuelve a velocidad normal si no se detecta el gesto
          }

        });
      }
    }
    frameCounter += 1;
    img.dispose();
    await tf.nextFrame();
  }
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
