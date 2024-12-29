import {pipeline} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.1.0";

const classifier = await pipeline('sentiment-analysis');

const output = await classifier('I love transformers!');

console.log(output);  // [{ label: 'POSITIVE', score: 0.9998 }]