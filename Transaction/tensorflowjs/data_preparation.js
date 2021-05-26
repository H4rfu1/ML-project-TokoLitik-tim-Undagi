const axios = require("axios");
const lstm_model = require("./lstm_model")

async function dataPreparation(productId, successCallback, errorCallback) {
  let result = []
  let productURL = `http://tokolitik.tech:3000/api/users/stores/products/${productId}/transactions/`;
  let token =
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOjEsImlhdCI6MTYyMTk5ODY0MiwiZXhwIjoxNjIyMDg1MDQyfQ.MyWeMnEm319LhRSabI61PLkSepOpbXZuVd6nYxzkP60";

  let headers = { "Content-Type": "application/json", "auth-token": token };

  await axios
    .get(productURL, { headers })
    .then((res) => {
      let resdata = {
        product: res.data.product,
        transaction: res.data.transaction
      }
      successCallback(resdata)
    })
    .catch((err) => {
      console.log("Error: ", err.message);
      errorCallback(err.message)
    });    
}

module.exports = {dataPreparation}

