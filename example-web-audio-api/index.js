const loadAudioBtn = document.querySelector("#loadAudioBtn");
const micStartBtn = document.querySelector("#micStartBtn");
const micStopBtn = document.querySelector("#micStopBtn");
const createAudioBtn = document.querySelector("#createAudioBtn");

const audioPlayer = document.querySelector("#audioPlayer");

const SAMPLE_RATE = 48000; // Frecuencia de muestreo
const NUM_SECONDS = 3; // Segundos de audio a procesar

let micAudioContext;
let micStream;
const timeDataQueue = [];

async function app() {
    micStopBtn.setAttribute("disabled", true);

    loadAudioBtn.onclick = async () => onLoadAudio("drop.wav");
    createAudioBtn.onclick = () => onCreateAudio(5000);
    micStartBtn.onclick = async () => onMicStart();
    micStopBtn.onclick = async () => onMicStop(); 
}

app();


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

    micStartBtn.disabled = true;
    micStopBtn.disabled = false;


    recorder.port.onmessage =  async(e) => {
        const inputBuffer = Array.from(e.data);
        
        if (inputBuffer[0] === 0) return;

        timeDataQueue.push(...inputBuffer);

        const num_samples = timeDataQueue.length;
        if (num_samples >= SAMPLE_RATE * NUM_SECONDS) {
            const audioData = new Float32Array(timeDataQueue.splice(0, SAMPLE_RATE * NUM_SECONDS));
            console.log("audioData", audioData);

            createAudioElementFromBuffer(audioData.buffer, SAMPLE_RATE);
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
        micStopBtn.disabled = true;
        micStartBtn.disabled = false;
    }

}

async function onLoadAudio(url) {
    const audioData = await getTimeDomainDataFromFile(url);
    console.log(audioData);
    playAudio(url);
}

function onCreateAudio(millis) {
    const audioData = new Float32Array(Array.from(
        {
            length: SAMPLE_RATE * millis / 1e3
        }, () => Math.random() * 2 - 1));

    const wavBytes = getWavBytes(audioData.buffer);

    const blob = new Blob([wavBytes], { 'type': 'audio/wav' });
    const audioURL = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = audioURL;
    a.download = "test.wav";
    a.click();
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

async function playAudio(url) {
    audioPlayer.src = url;
    audioPlayer.load();
    audioPlayer.onloadeddata = function () { audioPlayer.play(); };
}

async function getTimeDomainDataFromFile(url) {
    const audioContext = new AudioContext({
        sampleRate: SAMPLE_RATE
    });
    const response = await fetch(url);
    const arrayBuffer = await response.arrayBuffer();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    return audioBuffer.getChannelData(0);
}

function createAudioElementFromBuffer(audioBuffer) {  
    const wavBytes = getWavBytes(audioBuffer);
    const audioBlob = new Blob([wavBytes], { 'type': 'audio/wav' });
    const audioUrl = window.URL.createObjectURL(audioBlob);
    const audioElement = document.createElement("audio");
    audioElement.src = audioUrl;
    audioElement.controls = true;
    document.body.appendChild(audioElement);
  }


// Retorna Uint8Array de bytes WAV
function getWavBytes(buffer) {
    const numFrames = buffer.byteLength / Float32Array.BYTES_PER_ELEMENT;
    const headerBytes = getWavHeader(numFrames);
    const wavBytes = new Uint8Array(headerBytes.length + buffer.byteLength);

    // prepend header, then add pcmBytes
    wavBytes.set(headerBytes, 0);
    wavBytes.set(new Uint8Array(buffer), headerBytes.length);

    return wavBytes;
}

function getWavHeader(numFrames) {
    const numChannels = 1;
    const bytesPerSample = 4;

    const format = 3; //Float32

    const blockAlign = numChannels * bytesPerSample;
    const byteRate = SAMPLE_RATE * blockAlign;
    const dataSize = numFrames * blockAlign;

    const buffer = new ArrayBuffer(44);
    const dv = new DataView(buffer);

    let p = 0;

    function writeString(s) {
        for (let i = 0; i < s.length; i++) {
            dv.setUint8(p + i, s.charCodeAt(i));
        }
        p += s.length;
    }

    function writeUint32(d) {
        dv.setUint32(p, d, true);
        p += 4;
    }

    function writeUint16(d) {
        dv.setUint16(p, d, true);
        p += 2;
    }

    writeString('RIFF');              // ChunkID
    writeUint32(dataSize + 36);       // ChunkSize
    writeString('WAVE');              // Format
    writeString('fmt ');              // Subchunk1ID
    writeUint32(16);                  // Subchunk1Size
    writeUint16(format);              // AudioFormat
    writeUint16(numChannels);         // NumChannels
    writeUint32(SAMPLE_RATE);   // SampleRate
    writeUint32(byteRate);            // ByteRate
    writeUint16(blockAlign);          // BlockAlign
    writeUint16(bytesPerSample * 8);  // BitsPerSample
    writeString('data');              // Subchunk2ID
    writeUint32(dataSize);            // Subchunk2Size

    return new Uint8Array(buffer);
}


