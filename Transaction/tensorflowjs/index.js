const data_preparation = require("./data_preparation");
const model = require("./lstm_model");
let productId = 1;

function preparedData(productId) {
  return new Promise((resolve, reject) => {
    data_preparation.dataPreparation(
      productId,
      (successResponse) => {
        resolve(successResponse);
        return successResponse;
      },
      (errorResponse) => {
        reject(errorResponse);
      }
    );
  });
}

function preprocessedData(successCallback, errorCallback) {
  preparedData(productId)
    .then((res) => {
      let data = model.dataTransformation(res);
      successCallback(data)
    })
    .catch((e) => errorCallback(e));
}

function LSTM_Model() {
  return new Promise((resolve, reject) => {
    preprocessedData((successResponse) => {
        resolve(successResponse);
        model.getModel(successResponse)
      },
      (errorResponse) => {
        reject(errorResponse);
      }
    );
  });
}

LSTM_Model();
