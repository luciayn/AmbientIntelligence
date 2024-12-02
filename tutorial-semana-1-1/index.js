const outputMessageEl = document.querySelector("#outputMessage");
const StartModel = document.querySelector("#StartModel");

function initTFJS() {
  if (typeof tf === "undefined") {
    throw new Error("TensorFlow.js not loaded");
  }
}

async function startClassification() {

  console.log("Waiting for MobileNet to load...");

  if (outputMessageEl) {
    outputMessageEl.innerText = "Waiting for MobileNet to load...";
  }

  // Load the model.
  model = await mobilenet.load();  // método asíncrono, tenemos que esperar a que el modelo cargue
  console.log("MobileNet loaded successfully");

  if (outputMessageEl) {
    outputMessageEl.innerText = "MobileNet loaded successfully";
  }

  // Make a prediction with the model.
  const imgEl = document.querySelector("#img");
  const result = await model.classify(imgEl); // podemos pasar instancias en distintos formatos (mirar Github)
  // console.log(result);
  console.log(result[0].className);

  if (outputMessageEl) {
    outputMessageEl.innerText = result[0].className;
  }

}

async function app() {
  let model;
  StartModel.disabled = false;
  StartModel.addEventListener("click", startClassification)
  //await startClassification();
  // const obj = {}; // objeto vacío
  // obj.a = a;  // asignar valores

  // Using the promises syntax
  // mobilenet.load().then((model) => {
  //   console.log("MobileNet loaded successfully");
  //   const imgEl = document.querySelector("#img");
  //   model.classify(imgEl).then((result) => {
  //     console.log(result);
  //   });
  // });
  
}

(async function initApp() {

  try {
    initTFJS();
    await app();
  } catch (error) {
    console.error(error);
    if (outputMessageEl) {
      outputMessageEl.innerText = error.message;
    }
  }

}());


