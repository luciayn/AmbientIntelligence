import { TextStreamer, pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.1.0";

const TASK_NAME = "text-generation";
const MODEL_NAME = "onnx-community/Qwen2.5-0.5B-Instruct";

let generator = null;
let streamer = null;
let text = null;

let sound_type = null;
let location = null;
let urgency = null;
let contexto = null;

let output = null;

self.onmessage = async (e) => {
    console.log(e);
    switch (e.data.type) {
        // Cargar modelo
        case 'load':
            await load();
            break;

        // Generar texto relacionado con sonidos de alarmas
        case 'generate_alarm':
            sound_type = 'Alarma Sound';
            location = 'Platform Track 2';
            urgency = 'High';
            contexto = `An alarm is sounding on Platform Track 2. Generate a polite message to display on screens that reminds passengers not to enter the car once the door closing alarm has started.`;
            await generate();
            self.postMessage({ type: "translate"});
            break;
        
        // Generar texto relacionado con golpes de asientos
        case 'generate_seat':
            sound_type = 'Seat Bump';
            location = 'Carriage 5';
            urgency = 'Moderate';
            contexto = `A passenger has hit the seat when getting up in Carriage 5. Generate a polite message to display on screens reminding passengers to handle the seats with care.`;
            await generate();
            self.postMessage({ type: "translate"});
            break;
        
        // Generar texto relacionado con equipaje de ruedas en zonas no permitidas
        case 'generate_luggage':
            sound_type = 'Rolling Luggage in Non-Permitted Areas';
            location = 'Waiting Room 2';
            urgency = 'Low';
            contexto = `A passenger is dragging the suitcase in the Waiting Room 2. Generate a polite message to display on screens that reminds passengers to carry their suitcases in their hands or leave their luggage in the room located at the entrance of the building.`;
            await generate();
            self.postMessage({ type: "translate"});
            break;
        
        // Generar traducciones del texto generado en múltiples idiomas
        case 'start_translation':
            console.log('Initiating Translating System');
            await translateMessage(text);
            break;
    }

};

// Cargar el modelo
async function load() {
    
    generator = await pipeline(
        TASK_NAME,
        MODEL_NAME,
        { dtype: "fp16", device: "wasm", }
    );

    // Permite enviar los tokens a medida que el modelo los genera
    streamer = new TextStreamer(generator.tokenizer, {
        skip_prompt: true,
        callback_function,
    });

    // WARM-UP: Perform a dummy inference
    await generator("Warm up", {
        max_new_tokens: 1
    });

    self.postMessage({ type: "ready" });
}

// Generar texto dependiendo del tipo de sonido, ubicación, urgencia, y contexto
// Adjuntamos un ejemplo (one-shot prompting)
async function generate() {
    const prompt = `
    You are a worker in the public transport sector in charge of transmitting announcements to passengers.
    Your objective is to promote a more respectful and accessible environment, improving coexistence between passengers.  
    Your responsibility is to generate messages to display on screens next to the affected areas.
    You MUST NOT include any type of Python code nor add any "Note: ".

    Example 1:
    Sound Type: Seat Bumps
    Location: Carriage 5
    Urgency: Moderate
    Context: A passenger has hit the seat when getting up in Carriage 5. Generate a polite message to display on screens reminding passengers to handle the seats with care.
    Answer: "Please make sure to handle the seats carefully when getting up to avoid bumps or accidents. Thank you for your collaboration."

    Now, generate a message given the following characteristics:
    Sound Type: ${sound_type}
    Location: ${location}
    Urgency: ${urgency}
    Context: ${contexto}
    Answer:
    `;

    output = await generator(prompt, {
        max_new_tokens: 30, // Limitamos la generación a 30 tokens
        temperature: 0.2,  // Prevenir alucionaciones
        top_p: 0.5,
        do_sample: false,
        early_stopping: true,
        streamer
    });

    // Accedemos al texto generado
    text = output[0].generated_text.replace(prompt, '').trim();
    console.log(`output: ${output[0].generated_text.replace(prompt, '').trim()}`);
}

// Generar la traducción dado el texto generado (text) dependiendo del idioma deseado (targetLanguage)
// Adjuntamos varios ejemplos (multishot prompting)
async function generate_translation(text, targetLanguage) {
    const prompt = `
    You are a professional translator.
    The output MUST ONLY be the translated sentence.
    You MUST NOT include any type of Python code nor Notes.
    
    Example 1:
    Original Text: "Please make sure to handle the seats carefully when getting up to avoid bumps or accidents. Thank you for your collaboration."
    Translation in Spanish: "Por favor, asegúrese de manejar los asientos con cuidado al levantarse para evitar golpes o accidentes. Gracias por su colaboración."
    
    Example 2:
    Original Text: "Please make sure to handle the seats carefully when getting up to avoid bumps or accidents. Thank you for your collaboration."
    Translation in German: "Bitte achten Sie beim Aufstehen darauf, vorsichtig mit den Sitzen umzugehen, um Stöße oder Unfälle zu vermeiden. Vielen Dank für Ihre Zusammenarbeit."

    Translate the following text in ${targetLanguage}:
    Original Text: "${text}"
    Translation in ${targetLanguage}:
    `;

    const translatedText = await generator(prompt, {
        max_new_tokens: 35, // Limitamos la generación a 35 tokens
        temperature: 0.2, // Prevenir alucionaciones
        top_p: 0.5,
        do_sample: false,
        early_stopping: true
    });

    // Accedemos al texto generado
    console.log(`Translation in ${targetLanguage}: ${translatedText[0].generated_text.replace(prompt, '').trim()}`);
    return translatedText[0].generated_text.replace(prompt, '').trim();
}

async function translateMessage(text) {
    const languages = ['Spanish', 'German'];
    const translations = {};
    for (const lang of languages) {
        translations[lang] = await generate_translation(text, lang);

    }

    // Envío de las traducciones para su acceso en el archivo index.js
    self.postMessage({
        type: "translations",
        translations: {
            spanish: translations['Spanish'],
            german: translations['German']
        }
    });
}

// Envío de los tokens generados a medida que se crean
function callback_function(token) {
    self.postMessage({ type: "token", token });
}

