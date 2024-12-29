const btnMicStart = document.querySelector("#micStartBtn");
const btnMicStop = document.querySelector("#micStopBtn");

const SAMPLE_RATE = 16000; // Frecuencia de muestreo
const NUM_SECONDS = 3; // Segundos de audio a procesar

const YAMNET_MODEL_URL = 'https://www.kaggle.com/models/google/yamnet/TfJs/tfjs/1';

let micAudioContext;
let micStream;

let model;

const timeDataQueue = [];

async function onMicStart() {
    micStream = await getAudioStream();
    micAudioContext = new AudioContext({
        latencyHint: "playback",
        sampleRate: SAMPLE_RATE
    });

    const streamSource = micAudioContext.createMediaStreamSource(micStream);
    await micAudioContext.audioWorklet.addModule("recorder.worklet.js");
    const recorder = new AudioWorkletNode(micAudioContext, "recorder.worklet");
    streamSource.connect(recorder).connect(micAudioContext.destination);

    btnMicStart.disabled = true;
    btnMicStop.disabled = false;

    recorder.port.onmessage =  async(e) => {
        const inputBuffer = Array.from(e.data);
        
        if (inputBuffer[0] === 0) return;

        timeDataQueue.push(...inputBuffer);

        const num_samples = timeDataQueue.length;
        if (num_samples >= SAMPLE_RATE * NUM_SECONDS) {
            const audioData = new Float32Array(timeDataQueue.splice(0, SAMPLE_RATE * NUM_SECONDS));
            console.log("Start classification");
            const audioTensor = tf.tensor(audioData);
            const [scores, embeddings, spectrogram] = model.predict(audioTensor);
            scores.mean(axis=0).argMax().print(verbose=true);
        }
    }

};

async function onMicStop() {
    if (!Boolean(micAudioContext) || !Boolean(micStream)) return; 
    micAudioContext.close();
    micAudioContext = null;
    
    timeDataQueue.length = 0;
    if (micStream != null && micStream.getTracks().length > 0) {
        micStream.getTracks()[0].stop();
        btnMicStop.disabled = true;
        btnMicStart.disabled = false;
    }
}

async function getAudioStream(audioTrackConstraints) {
    let options = audioTrackConstraints || {};
    try {
        return await navigator.mediaDevices.getUserMedia({
            video: false,
            audio: {
                sampleRate: options.sampleRate || SAMPLE_RATE,
                sampleSize: options.sampleSize || 16,
                channelCount: options.channelCount || 1
            }
        });
    } catch (e) {
        console.error(e);
        return null;
    }
}

function initTFJS() {
    if (typeof tf === "undefined") {
      throw new Error("TensorFlow.js not loaded");
    }
  }

async function app() {  
    btnMicStart.disabled = true;
    btnMicStop.disabled = true;
    
    model = await tf.loadGraphModel(YAMNET_MODEL_URL, { fromTFHub: true });
    console.log("YAMNet model loaded:");

    btnMicStart.disabled = false;
    btnMicStart.onclick = async () => onMicStart();
    btnMicStop.onclick = async () => onMicStop();
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


