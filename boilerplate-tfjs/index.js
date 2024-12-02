const outputMessageEl = document.querySelector("#outputMessage");

function initTFJS() {
  if (typeof tf === "undefined") {
    throw new Error("TensorFlow.js not loaded");
  }
}

// Si no tengo que esperar el resultado
async function app() { // asíncrona (ejecución a la misma vez que otro) para que no bloquee/nos carguemos el flujo de la página
  // Application code here
  if (outputMessageEl) {
    outputMessageEl.innerText = "TensorFlow.js version " + tf.version.tfjs;
  }
}

// await app(); // si lo ponemos fuera, tenemos un error (mirar consola en devtools)

// Si tengo que esperar el resultado
(async function initApp() {

  try {
    initTFJS();
    await app(); // esperar el resultado de una applicación asíncrona, solo podemos poner await dentro de una función async
  } catch (error) {
    console.error(error);
    if (outputMessageEl) {
      outputMessageEl.innerText = error.message;
    }
  }

}());  // (): nos permite encapsular el código asíncrono y ejecutarlo de forma automática
