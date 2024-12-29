import {pipeline} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.1.0";

// Create a text generation pipeline
const generator = await pipeline( "text-generation", "onnx-community/Qwen2.5-0.5B-Instruct",
    { dtype: "q4", device: "webgpu" },
  );
  
  // Define text
  const message = "Once upon a time, "

  // Generate a response
  let output = await generator(message, { 
    max_new_tokens: 128,
    temperature: 0.9,
    repetition_penalty: 2.0,
    no_repeat_ngram_size: 3
});

console.log(output[0].generated_text);