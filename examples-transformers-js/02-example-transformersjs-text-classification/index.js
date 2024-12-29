import {pipeline} from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.1.0";

const classifier = await pipeline('text-classification', 'Xenova/toxic-bert');

const output = await classifier('I love I hate you!', {top_k:null});

console.log(output); 
// [   
//     { "label": "toxic","score": 0.8899049758911133 },
//     { "label": "insult", "score": 0.10198599100112915 },
//     { "label": "identity_hate", "score": 0.04956541955471039 },
//     { "label": "obscene", "score": 0.01883940026164055 },
//     { "label": "threat", "score": 0.018221747130155563 },
//     { "label": "severe_toxic", "score": 0.003908987622708082 }
// ]

