const webcamEl = document.querySelector("#webcam");
const liveView = document.querySelector("#liveView");
const appSection = document.querySelector("#app");
const btnEnableWebcam = document.querySelector("#btnEnableWebcam");
const outputMessageEl = document.querySelector("#outputMessage");
const history_list = document.querySelector("#history_list");

let webcam;
let model;

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
  try {
    webcam = await tf.data.webcam(webcamEl);
    outputMessageEl.innerText = "Webcam enabled! Detecting objects...";
    await predictWebcam();
  } catch (error) {
    console.error("Webcam access failed:", error);
    outputMessageEl.innerText = "Webcam access denied.";
  }
}

async function loadCocoSsdModel() {
  model = await cocoSsd.load();
  btnEnableWebcam.disabled = false;
  outputMessageEl.innerText = "Model loaded! Enable the webcam to start detection.";
}


async function predictWebcam() {
  // Array to store created DOM elements for object annotations
  const objects = [];

  while (true) {
     // Capture the current frame from the webcam
    const frame = await webcam.capture();
    const predictions = await model.detect(frame);

    // Remove previously created DOM elements from the live view
    objects.forEach((object) => liveView.removeChild(object));
    objects.length = 0;
    for (let n = 0; n < predictions.length; n++) {
      // Only process predictions with a confidence score above 66%
      if (predictions[n].score > 0.66) {
        // Create a paragraph element to display the class and confidence
        const p = document.createElement("p");
        const c = predictions[n].class;
        const score = Math.round(parseFloat(predictions[n].score) * 100);
        //if (c == "bottle") {
        p.innerText = `${c} - with ${score}% confidence.`;
        p.style = `margin-left: ${predictions[n].bbox[0]}px; 
                  margin-top: ${predictions[n].bbox[1] - 10}px; 
                  width: ${predictions[n].bbox[2] - 10}px; 
                  top: 0; 
                  left: 0;`;
                  
        // Create a highlighter div to visually outline the detected object
        const highlighter = document.createElement("div");
        highlighter.setAttribute("class", "highlighter");
        highlighter.style = `left: ${predictions[n].bbox[0]}px; 
                            top: ${predictions[n].bbox[1]}px; 
                            width: ${predictions[n].bbox[2]}px; 
                            height: ${predictions[n].bbox[3]}px;`;
        if (history_list.children.length > 3) {
          history_list.removeChild(history_list.firstChild);
        }
        const history = document.createElement("div");
        history.innerText = `${c} - with ${score}% confidence.`;
        history_list.appendChild(history);
        liveView.appendChild(highlighter);
        liveView.appendChild(p);
        objects.push(highlighter);
        objects.push(p);
        //}
      }
    }
    frame.dispose();
    await tf.nextFrame();
  }
}

async function app() {
  // Check for webcam support before proceeding
  if (!getUserMediaSupported()) {
    console.warn("getUserMedia() is not supported by your browser");
    outputMessageEl.innerText = "Webcam not supported by your browser.";
    return; // Exit if webcam is not supported
  }
  btnEnableWebcam.addEventListener("click", enableCam);

  await loadCocoSsdModel(); // Load the model if webcam support is confirmed
  
}

(async function initApp() {

  try {
    initTFJS();
    await app();
  } catch (error) {
    console.error(error);
    outputMessageEl.innerText = error.message;
  }

}());



