import {pipeline} from "https://cdn.jsdelivr.net/npm/@xenova/transformers";

// Create a text generation pipeline
const generator = await pipeline( "text-generation", "Xenova/gpt2");
  
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

