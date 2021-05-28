const { spawn } = require("child_process");

function prepareData(url, token, successCallback, errorCallback) {
  const dataPreparation = spawn("python", [
    "dataset_preparation.py",
    url,
    token,
  ]);

  dataPreparation.stdout.on("data", (auth_status) => {
    successCallback(`${auth_status}`);
  });

  dataPreparation.stderr.on("data", (data) => {
    errorCallback(data);
  });

  dataPreparation.on("close", (code) => {
    console.log(`child process exited with code ${code}`);
  });
}

function modelPredict(id, url, token) {
  return new Promise((resolve, reject) => {
    prepareData(
      url,
      token,
      (auth_status) => {
        resolve(auth_status);
        if (auth_status == 200) {
          const modelData = spawn("python", ["model_predict.py", id]);
          modelData.stdout.on("data", (data) => {
            console.log(`${data}`)
            return;
          });

          modelData.stderr.on("data", (data) => {
            return
          });

          modelData.on("close", (code) => {
            console.log(`child process exited with code ${code}`);
          });
        } else {
          console.log("Authentication failed");
        }
      },
      (err) => {
        reject(err);
      }
    );
  });
}

let productId = 1;
const token =
  "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOjEsImlhdCI6MTYyMjE3Mzg5NCwiZXhwIjoxNjIyMjYwMjk0fQ.2uOInISldLxYy5Ao8RhICYoCfgATgkn3T2y8QbZreSs";
const URL = `http://tokolitik.tech:3000/api/users/stores/products/${productId}/transactions/`;
modelPredict(productId, URL, token);
