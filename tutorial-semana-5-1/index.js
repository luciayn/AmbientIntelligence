const outputMessageEl = document.getElementById('outputMessage');

const worker = new Worker('worker.js', { type: "module" }); // Path to your worker file
worker.postMessage({ type: 'load' });

worker.onmessage = async (e) => {
  switch(e.data.type) {
    case 'token':
      console.log(e.data.token);
      outputMessageEl.textContent += e.data.token;
      break;
      
    case 'ready':
      console.log('Ready');
      worker.postMessage({ type: 'generate' });
      break;
  }
};