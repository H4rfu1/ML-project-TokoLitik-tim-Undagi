const moment = require("moment");
const MinMaxScaler = require("minmaxscaler")
// const tf = require('@tensorflow/tfjs-node');
// const { model } = require("@tensorflow/tfjs-node");

function dataPreprocessing(data) {
  let transaction = data.transaction;
  let product = data.product;

  let new_data = [];

  // change data format from YYYY-MM-DD to YYYY-MM
  for (let i in transaction) {
    let data_tmp = transaction[i];
    data_tmp.time = moment(data_tmp.time).format("YYYY-MM") + "-01";
    new_data.push(data_tmp);
  }

  // accumulate daily data into monthly data
  let result = Object.values(
    new_data.reduce((a, { id, productId, time, amount }) => {
      a[time] = a[time] || { id, productId, time, amount: 0 };
      a[time].amount = String(Number(a[time].amount) + Number(amount));
      return a;
    }, {})
  );

  return result;
}

function dataTransformation(data) {
  let monthly_dataset = dataPreprocessing(data);
  let monthly_sales_data = monthly_dataset.map((data) => data.amount);

  let index = 0;
  let x_data = [];
  let y_data = [];

  while (index < monthly_sales_data.length) {
    if (index + 6 <= monthly_sales_data.length - 6) {
      // let x_temp = monthly_sales_data.slice(index, index + 6);
      
      let x_temp = monthly_sales_data.slice(index, index+6).map(i => [i])
      let y_temp = monthly_sales_data.slice(index+6, index+12)
      x_data.push([x_temp]);
      y_data.push(y_temp)
    }

    index++;
  }
  
  return {x_data: x_data, y_data: y_data};
}


function getModel(){
  // const model = tf.sequential()
  // model.add(tf.layers.bidirectional(tf.layers.lstm({units: 100, activation:"relu", returnSequences:true, inputShape:(6,1)})))
  // model.add(tf.layers.lstm({units:100, activation:"relu"}))
  // model.add(tf.layers.dense({units:6}))

  // model.compile({optimizer:"adam", loss:"mse"})

  console.log("model called")
  // return model
}

function trainModel(data){
  // let model = getModel()
  // return model.fit(data.x_data, data.y_data, {
  //   epochs:300,
  //   verbose:1
  // })
}


// module exports
module.exports = { dataTransformation, getModel, trainModel};
