const btnCreateAndTrain = document.querySelector("#btnCreateAndTrain");
const btnTestModel = document.querySelector("#btnTestModel");
const btnLoadModel = document.querySelector("#btnLoadModel");
const btnMicStart = document.querySelector("#btnMicStart");
const btnMicStop = document.querySelector("#btnMicStop");
const btnCreateSampleAudio = document.querySelector("#btnCreateSampleAudio");

const audioPlayer = document.querySelector("#audioPlayer");

/* El código está escrito así con fines educativos. 
 * No es el código que usaríamos en producción
 */
const YAMNET_MODEL_URL = "https://tfhub.dev/google/tfjs-model/yamnet/tfjs/1";

/* 
 * Parámetros para la creación del modelo
 */
const INPUT_SHAPE = 1024;
const NUM_CLASSES = 4;

/* 
 * Parámetros para el procesamiento de audio
 */
const MODEL_SAMPLE_RATE = 16000; // Frecuencia de muestreo para YAMNet
const NUM_SECONDS = 3; // Número de segundos para el muestreo desde mic

const CLASSES = ["crying_baby", "clock_alarm", "toilet_flush", "water_drops"];


function flattenQueue(queue) {
    const frameSize = queue[0].length;
    const data = new Float32Array(queue.length * frameSize);
    queue.forEach((d, i) => data.set(d, i * frameSize));
    return data;
}

let model;
let yamnet;

function initTFJS() {
    if (typeof tf === "undefined") {
        throw new Error("TensorFlow.js not loaded");
    }
}

async function loadCsvMetadata(csvUrl) {
    const metadata = tf.data.csv(csvUrl, {
        hasHeader: true
    });

    return await metadata.toArray();;
}

async function app() {
    let audioContext;
    let stream;

    const timeDataQueue = [];


    const trainCsvUrl = "data/trainMetadata.csv"; // Path to your training CSV
    const testCsvUrl = "data/testMetadata.csv";  // Path to your testing CSV

    const trainDataArray = await loadCsvMetadata(trainCsvUrl);
    const testDataArray = await loadCsvMetadata(testCsvUrl);

    enableAllButtons(false);
    enableButton(btnCreateSampleAudio, true);

    btnCreateSampleAudio.onclick = async () => {
        createAudio(5000);
    };

    yamnet = await loadYamnetModel();
    console.log("YamNet model loaded");
    enableButton(btnCreateAndTrain, true);
    enableButton(btnLoadModel, true);

    enableButton(btnTestModel, false);
    enableButton(btnMicStart, false);
    enableButton(btnMicStop, false);

    btnMicStart.onclick = async () => {

        stream = await getAudioStream();
        audioContext = new AudioContext({
            latencyHint: "playback",
            sampleRate: MODEL_SAMPLE_RATE
        });

        const streamSource = audioContext.createMediaStreamSource(stream);

        await audioContext.audioWorklet.addModule("recorder.worklet.js");
        const recorder = new AudioWorkletNode(audioContext, "recorder.worklet");
        streamSource.connect(recorder).connect(audioContext.destination);

        enableButton(btnMicStart, false);
        enableButton(btnMicStop, true);

        recorder.port.onmessage = async (e) => {
            const inputBuffer = Array.from(e.data);

            if (inputBuffer[0] === 0) return;

            timeDataQueue.push(...inputBuffer);

            const num_samples = timeDataQueue.length;
            if (num_samples >= MODEL_SAMPLE_RATE * NUM_SECONDS) {
                const audioData = new Float32Array(timeDataQueue.splice(0, MODEL_SAMPLE_RATE * NUM_SECONDS));
                console.log(CLASSES[await predict(yamnet, model, audioData)]);
            }
        }
    };

    btnMicStop.onclick = () => {
        if (!Boolean(audioContext) || !Boolean(stream)) return;
        audioContext.close();
        audioContext = null;

        timeDataQueue.splice(0);
        if (stream != null && stream.getTracks().length > 0) {
            stream.getTracks()[0].stop();
            enableButton(btnMicStart, true);
            enableButton(btnMicStop, false);
        }
    }


    btnLoadModel.onclick = async () => {
        model = await loadCustomAudioClassificationModelFromFile("./model/model.json");
        enableButton(btnTestModel, true);
        enableButton(btnMicStart, true);
        enableButton(btnCreateAndTrain, false);
        enableButton(btnLoadModel, false);
    }

    btnTestModel.onclick = async () => {
        testCustomAudioClassificationModel(yamnet, model, testDataArray);
    };

    btnCreateAndTrain.onclick = async () => {
        enableAllButtons(false);
        model = await createAndTrainCustomAudioClassificationModel(yamnet, trainDataArray);
        enableAllButtons(true);
        enableButton(btnCreateAndTrain, false);
        enableButton(btnLoadModel, false);
    };

}


(async function initApp() {

    try {
        initTFJS();
        await app();

    } catch (error) {
        console.error(error);

    }

}());

async function loadYamnetModel() {
    const model = await tf.loadGraphModel(YAMNET_MODEL_URL, { fromTFHub: true });
    return model;
}

async function testCustomAudioClassificationModel(yamnet, model, testDataArray) {
    const RANDOM = Math.floor((Math.random() * testDataArray.length));
    const testSample = testDataArray[RANDOM];
    // console.log(testSample);
    const audioData = await getTimeDomainDataFromFile(`data/audio/${testSample.fileName}`);
    playAudio(`data/audio/${testSample.fileName}`);
    const prediction = await predict(yamnet, model, audioData);
    console.log(CLASSES[prediction]);
}

async function predict(yamnet, model, audioData) {
    const embeddings = await getEmbeddingsFromTimeDomainData(yamnet, audioData);
    // embeddings.print(true);
    const results = await model.predict(embeddings);
    results.print(true)
    const meanTensor = results.mean((axis = 0));
    // meanTensor.print();
    const argMaxTensor = meanTensor.argMax(0);

    embeddings.dispose();
    results.dispose();
    meanTensor.dispose();
    return argMaxTensor.dataSync()[0];
}

async function loadCustomAudioClassificationModelFromFile(url) {
    const model = await tf.loadLayersModel(url);
    model.summary();
    return model;
}

function logProgress(epoch, logs) {
    console.log(`Data for epoch ${epoch}, ${Math.sqrt(logs.loss)}`);
}

async function prepareTrainingData(yamnet, metadata, audioBasePath) {
    const INPUT_DATA = [];
    const OUTPUT_DATA = [];

    for (const { fileName, classNumber } of metadata) {
        const audioData = await getTimeDomainDataFromFile(`${audioBasePath}/${fileName}`);
        const embeddings = await getEmbeddingsFromTimeDomainData(yamnet, audioData);
        const embeddingsArray = embeddings.arraySync();

        for (const embedding of embeddingsArray) {
            INPUT_DATA.push(embedding);
            OUTPUT_DATA.push(classNumber);
        }

        embeddings.dispose();
    }

    tf.util.shuffleCombo(INPUT_DATA, OUTPUT_DATA);

    const inputTensor = tf.tensor2d(INPUT_DATA);
    const outputAsOneHotTensor = tf.oneHot(tf.tensor1d(OUTPUT_DATA, 'int32'), NUM_CLASSES);

    return [inputTensor, outputAsOneHotTensor];
}

async function createAndTrainCustomAudioClassificationModel(yamnet, trainDataArray) {

    const [inputTensor, outputAsOneHotTensor] = await prepareTrainingData(yamnet, trainDataArray, "data/audio");
    // outputAsOneHotTensor.print(true);

    const model = createModel();
    await trainModel(model, inputTensor, outputAsOneHotTensor);
    await saveModel(model);
    return model;
}

async function saveModel(model) {
    model.save('downloads://model');
}

function createModel() {
    const model = tf.sequential();

    model.add(tf.layers.dense({ dtype: 'float32', inputShape: [INPUT_SHAPE], units: 512, activation: 'relu' }));
    model.add(tf.layers.dense({ units: NUM_CLASSES, activation: 'softmax' }));
    model.summary();
    return model;
}

async function trainModel(model, inputTensor, outputTensor) {
    model.compile({
        optimizer: 'adam',
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    const params = {
        shuffle: true,
        validationSplit: 0.15,
        batchSize: 16,
        epochs: 20,
        callbacks: [new tf.CustomCallback({ onEpochEnd: logProgress }),
            //tf.callbacks.earlyStopping({ monitor: 'loss', patience: 3 })
        ]
    };

    const results = await model.fit(inputTensor, outputTensor, params);
    console.log("Average error loss: " + Math.sqrt(results.history.loss[results.history.loss.length - 1]));
    console.log("Average validation error loss: " +
        Math.sqrt(results.history.val_loss[results.history.val_loss.length - 1]));
}

async function getAudioStream(audioTrackConstraints) {
    let options = audioTrackConstraints || {};
    try {
        return await navigator.mediaDevices.getUserMedia({
            video: false,
            audio: {
                sampleRate: options.sampleRate || MODEL_SAMPLE_RATE,
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

// Mejor crear un único AudioContext para todo el proceso
async function getTimeDomainDataFromFile(url) {
    const audioContext = new AudioContext({
        playbackLatencyHint: "playback",
        sampleRate: MODEL_SAMPLE_RATE
    });
    const response = await fetch(url);
    const arrayBuffer = await response.arrayBuffer();

    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
    audioContext.close();
    return audioBuffer.getChannelData(0);
}

async function getEmbeddingsFromTimeDomainData(model, audioData) {
    const waveformTensor = tf.tensor(audioData);
    // waveformTensor.print(true);
    const [scores, embeddings, spectrogram] = model.predict(waveformTensor);

    if (visualizationEnabled) visualizeData(audioData, spectrogram);

    waveformTensor.dispose();
    return embeddings;
}

function enableButton(buttonElement, enabled) {
    buttonElement.disabled = !enabled;
}

function enableAllButtons(enabled) {
    document.querySelectorAll("button").forEach(btn => {
        btn.disabled = !enabled;
    });
}


function createAudio(millis) {
    const audioData = new Float32Array(Array.from(
        {
            length: MODEL_SAMPLE_RATE * millis / 1e3
        }, () => Math.random() * 2 - 1));
 
 
    const wavBytes = getWavBytes(audioData.buffer);
 
 
    const blob = new Blob([wavBytes], { 'type': 'audio/wav' });
    const audioURL = window.URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = audioURL;
    a.download = "test.wav";
    a.click();
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
    const byteRate = MODEL_SAMPLE_RATE * blockAlign;
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
    writeUint32(MODEL_SAMPLE_RATE);   // SampleRate
    writeUint32(byteRate);            // ByteRate
    writeUint16(blockAlign);          // BlockAlign
    writeUint16(bytesPerSample * 8);  // BitsPerSample
    writeString('data');              // Subchunk2ID
    writeUint32(dataSize);            // Subchunk2Size

    return new Uint8Array(buffer);
}

const btnToggleViz = document.querySelector("#btnToggleViz");
btnToggleViz.addEventListener("click", () => {
    tfvis.visor().toggle();
});

let visualizationEnabled = false;

const checkViz = document.querySelector("#checkViz");
checkViz.addEventListener("change", () => {
    visualizationEnabled = checkViz.checked;
    enableButton(btnToggleViz, visualizationEnabled);
});

async function visualizeData(audioData, spectrogram) {
    // Visualize the waveform
    enableButton(btnToggleViz, true);  
    tfvis.render.linechart(
        { name: 'Waveform' },
        { values: Array.from(audioData).map((y, x) => ({ x, y })) },
        { width: 600, height: 200 }
    );

    // Visualize the spectrogram
    const spectrogramArray = spectrogram.arraySync();
    console.log(spectrogram.shape);
    const xTickLabels = Array.from({ length: spectrogramArray.length }, (_, i) => `${i}`);
    const yTickLabels = Array.from({ length: 64 }, (_, i) => `${64- i}`);
    tfvis.render.heatmap(
        { name: 'Mel Spectrogram' },
        { values: spectrogramArray, xTickLabels: xTickLabels, yTickLabels: yTickLabels},
        { width: 600, height: 400, xLabel: 'Time frames', yLabel: 'Mel bins', domain: [Math.min(...spectrogramArray.flat()), Math.max(...spectrogramArray.flat())] }
    );
}


