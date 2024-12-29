const WHISPER_SAMPLING_RATE = 16000;
const MAX_AUDIO_LENGTH = 5; // seconds

const MAX_SAMPLES = WHISPER_SAMPLING_RATE * MAX_AUDIO_LENGTH;

let language = 'en';

const timeDataQueue = [];

const btnStart = document.querySelector('#btnStart');
const btnStop = document.querySelector('#btnStop');
const outputMessageEl = document.querySelector('#outputMessage');

let audioContext;
let stream;

(async function app() {
    if (navigator.mediaDevices.getUserMedia) {
        
        btnStart.onclick = async () => {
            btnStart.disabled = true;
            btnStop.disabled = false;
            stream = await getAudioStream();
            audioContext = new AudioContext({
                latencyHint: "playback",
                sampleRate: WHISPER_SAMPLING_RATE
            });
            const streamSource = audioContext.createMediaStreamSource(stream);

            await audioContext.audioWorklet.addModule("recorder.worklet.js");
            const recorder = new AudioWorkletNode(audioContext, "recorder.worklet");
            streamSource.connect(recorder).connect(audioContext.destination);
            console.log("Start recording...");

            recorder.port.onmessage = async (e) => {
                const inputBuffer = Array.from(e.data);
                if (inputBuffer[0] === 0) return;

                timeDataQueue.push(...inputBuffer);

                if (timeDataQueue.length >= MAX_SAMPLES) {
                    const audioData = new Float32Array(timeDataQueue.splice(0, MAX_SAMPLES));
                    worker.postMessage({ type: 'generate', data: { audio: audioData, language } });

                }
            }
        };

        btnStop.onclick = () => {
            btnStart.disabled = false;
            btnStop.disabled = true;
            if (!Boolean(audioContext) || !Boolean(stream)) return;
            audioContext.close();

            timeDataQueue.length = 0;
            if (stream != null && stream.getTracks().length > 0) {
                stream.getTracks()[0].stop();
            }
        }


    }
}());

async function getAudioStream(audioTrackConstraints) {
    let options = audioTrackConstraints || {};
    try {
        return await navigator.mediaDevices.getUserMedia({
            video: false,
            audio: {
                sampleRate: options.sampleRate || WHISPER_SAMPLING_RATE,
                sampleSize: options.sampleSize || 16,
                channelCount: options.channelCount || 1
            }
        });
    } catch (e) {
        console.error(e);
        return null;
    }
}


const worker = new Worker('whisper.worker.js', { type: "module" }); // Path to your worker file
// Set up event listener for messages from the worker
worker.onmessage = function (e) {
    switch (e.data.status) {
        case 'loading':
            console.log('Loading status:', e.data.data);
            break;

        case 'initiate':
            break;

        case 'progress':
            break;

        case 'done':
            break;

        case 'ready':
            console.log('Worker is ready for processing.');
            btnStart.disabled = false;
            break;

        case 'start':
            break;

        case 'update':
            console.log(e.data.output);
            break;

        case 'complete':
            console.log(e.data.output);
            outputMessageEl.textContent = e.data.output[0];
            break;
        default:
            //console.error('Unknown status:', status);
    }
};

// Handle errors from the worker
worker.onerror = function (error) {
    console.error('Worker error:', error.message);
};

worker.postMessage({ type: 'load' });