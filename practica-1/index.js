const webcamEl = document.querySelector("#webcam");
const liveView = document.querySelector("#liveView");
const appSection = document.querySelector("#app");
const btnEnableWebcam = document.querySelector("#btnEnableWebcam");
const outputMessageEl = document.querySelector("#outputMessage");
const outputMessageEl1 = document.querySelector("#outputMessage1");
const outputMessageEl2 = document.querySelector("#outputMessage2");
const outputMessageEl3 = document.querySelector("#outputMessage3");
const history_list = document.querySelector("#history_list");

let webcam;
let model;
let knife_timestamp;

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
  const c_list = [];

  while (true) {
     // Capture the current frame from the webcam
    const frame = await webcam.capture();
    const predictions = await model.detect(frame);

    // Remove previously created DOM elements from the live view
    objects.forEach((object) => liveView.removeChild(object));
    objects.length = 0;
    c_list.length = 0;
    for (let n = 0; n < predictions.length; n++) {
      // Only process predictions with a confidence score above 1%
      if (predictions[n].score > 0.01) {
        // Create a paragraph element to display the class and confidence
        const p = document.createElement("p");
        const c = predictions[n].class;
        
        const score = Math.round(parseFloat(predictions[n].score) * 100);
        if (["bottle", "knife", "cup", "plate"].includes(c)) { // añadir "laptop" a la lista de objetos para la funcionalidad 3
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
          if (c == "laptop") {
            laptop_box = predictions[n].bbox;
          }
          if (c == "cup") {
            cup_box = predictions[n].bbox;
          }
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
          c_list.push(c);

          if (c_list.includes("bottle") && c_list.includes("cup")){
            outputMessageEl2.innerText = "¿Te gustaría tomar algo?";
          }
          else {
            outputMessageEl2.innerText = "\n"
          }

          if (c_list.includes("cup") && c_list.includes("laptop")){
            const proximityThreshold = 50;
            const cup_right = cup_box[0] + cup_box[2]; // Right side of the cup
            const cup_left = cup_box[0];              // Left side of the cup
            const laptop_left = laptop_box[0];         // Left side of the laptop
            const laptop_right = laptop_box[0] + laptop_box[2]; // Right side of the laptop
            const cup_bottom = cup_box[1] + cup_box[3]; // Bottom side of the cup
            const cup_top = cup_box[1];                 // Top side of the cup
            const laptop_top = laptop_box[1];           // Top side of the laptop
            const laptop_bottom = laptop_box[1] + laptop_box[3]; // Bottom side of the laptop

            if (Math.abs(cup_right - laptop_left) < proximityThreshold ||
            Math.abs(laptop_right - cup_left) < proximityThreshold ||
            Math.abs(cup_bottom - laptop_top) < proximityThreshold ||
            Math.abs(laptop_bottom - cup_top) < proximityThreshold) {
              outputMessageEl3.innerText = "¡Cuidado! La taza puede verterse sobre el ordenador.";
            }            
          }
          else {
            outputMessageEl3.innerText = "\n";
          }
        }
        if (c_list.includes("knife")){
          if (!knife_timestamp){
            knife_timestamp = Date.now();
          }
          else{
            elapsed_time = (Date.now()-knife_timestamp)/1000;
            outputMessageEl1.innerText = `Elapsed time: ${elapsed_time}`;
            if (elapsed_time > 60) {
              outputMessageEl1.innerText = "Recuerda guardar el cuchillo después de usarlo."
            }
          }
        }
        else {
          knife_timestamp = null;
        }
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



