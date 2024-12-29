const outputMessageEl = document.querySelector("#outputMessage");
const btnListen = document.querySelector("#btnListen");
let recognizer;

let isListening = false;


function initTFJS() {
  if (typeof tf === "undefined") {
    throw new Error("TensorFlow.js not loaded");
  }
}

async function app() {
  try {
    recognizer = speechCommands.create("BROWSER_FFT");
    console.log(recognizer.params());
    await recognizer.ensureModelLoaded();
    console.log("Modelo cargado con éxito");
    console.log(recognizer.wordLabels());

    btnListen.addEventListener("click", async () => {
      isListening = !isListening;
      if (isListening) {
        btnListen.innerText = "Stop Listening";
        recognizer.listen(result => {
          // - result.scores contains the probability scores that correspond to
          //   recognizer.wordLabels().
          // - result.spectrogram contains the spectrogram of the recognized word.
          const maxIndex = [...result.scores].reduce((maxIdx, currentValue, currentIndex, array) =>
            currentValue > array[maxIdx] ? currentIndex : maxIdx, 0);
        
          console.log(maxIndex);
        }, {
          includeSpectrogram: false,
          probabilityThreshold: 0.75
        });
      } else {
        btnListen.innerText = "Start Listening";
        recognizer.stopListening();
      }
      
    });

    btnListen.disabled = false;
  } catch (e) {
    console.log(e);
    return;
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




