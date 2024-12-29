import { TextStreamer, pipeline } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.1.0";

const TASK_NAME = "text-generation";
const MODEL_NAME = "onnx-community/Qwen2.5-0.5B-Instruct";

let generator = null;
let streamer = null;

self.onmessage = async (e) => {
    console.log(e);
    switch (e.data.type) {
        case 'load':
            await load();
            break;
        case 'generate':
            await generate();
            break;
    }

};

async function load() {
    
    generator = await pipeline(
        TASK_NAME,
        MODEL_NAME,
        { dtype: "fp16", device: "wasm", }
    );

    streamer = new TextStreamer(generator.tokenizer, {
        skip_prompt: false,
        callback_function,
    });

    // WARM-UP: Perform a dummy inference
    await generator("Warm up", {
        max_new_tokens: 1
    });

    self.postMessage({ type: "ready" });
}


async function generate() {
    const prompt = `You are an English tutor. Respond to students based on their language level and the specific topic of interest they mention. Always include examples or explanations directly related to the topic.

Example 1:
Student level: Beginner
Topic: Food
Student: "How do I order food in English?"
Tutor: "You can say: 'Can I have a pizza, please?' or 'I would like a burger, please.' Practice saying these phrases!"

Example 2:
Student level: Intermediate
Topic: Travel
Student: "What do I say when I need to find my gate at the airport?"
Tutor: "You can ask: 'Excuse me, where is gate 25?' or 'Can you help me find my gate, please?' These phrases are useful at airports."

Now, respond to this student:
Student level: Beginner
Topic: Food
Student: "How can I start a conversation with locals when traveling abroad?"
Tutor:
`;

    const output = await generator(prompt, {
        max_new_tokens: 64,
        //temperature: 0.5,
        top_p: 0.5,
        do_sample: false,
        early_stopping: true,
        streamer
    });

    console.log(output);
}

function callback_function(token) {
    self.postMessage({ type: "token", token });
}



