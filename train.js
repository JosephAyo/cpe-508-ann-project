const tf = require("@tensorflow/tfjs-node");
const createModel = require("./model2");
const createDataset = require("./normalizing/data2");

const csvPath = "./data/cleanedData.csv";

// train the model against the training data
const trainModel = async function (model, trainingData, epochs) {
  const options = {
    epochs: epochs,
    verbose: 0,
    // callbacks: {
    //   onEpochBegin: async (epoch, logs) => {
    //     console.log(`Epoch ${epoch + 1} of ${epochs} ...`);
    //   },
    //   onEpochEnd: async (epoch, logs) => {
    //     console.log(`  train-set loss: ${logs.loss.toFixed(4)}`);
    //     console.log(`  train-set accuracy: ${logs.acc.toFixed(4)}`);
    //   },
    // },
  };

  return await model.fitDataset(trainingData, options);
};

// verify the model against the test data
const evaluateModel = async function (model, testingData) {
  const result = await model.evaluateDataset(testingData);
  const testLoss = result[0].dataSync()[0];
  const testAcc = result[1].dataSync()[0];

  console.log(`  test-set loss: ${testLoss.toFixed(4)}`);
  console.log(`  test-set accuracy: ${testAcc.toFixed(4)}`);
};

// run
const run = async function (epochs, batchSize, savePath) {
  try {
    const datasetObj = await createDataset("file://" + csvPath);
    const model = createModel([datasetObj.numOfColumns]);
    model.summary();

    const trainBatches = Math.floor(50 / batchSize);
    const dataset = datasetObj.dataset.shuffle(1).batch(batchSize);
    const trainDataset = dataset.take(trainBatches);
    const validationDataset = dataset.skip(trainBatches);

    await model.fitDataset(trainDataset, {
      epochs: epochs,
      validationData: validationDataset,
    });

    await model.save(savePath);

    const info = await trainModel(model, trainDataset, epochs);
    console.log("\r\nInfo:", info);

    console.log("\r\nEvaluating model...");
    await evaluateModel(model, validationDataset);

    console.log("\r\nSaving model...");
    await model.save(savePath);

    // const loadedModel = await tf.loadLayersModel(savePath + "/model.json");
    // const result = loadedModel.predict(tf.tensor2d([[0, 0, 0.5]]));
    // console.log(
    //   "The actual value is 0, the inference result from the model is " +
    //     result.dataSync()
    // );

      // const loadedModel = await tf.loadLayersModel(savePath + "/model.json");
      // const result = loadedModel.predict(
      //   tf.tensor2d([[0.78, 0.98, 0.75, 0.89, 1.262, 0.507, 0.318, 0.39]])
      // );
      // console.log(
      //   "The actual test abalone age is 10, the inference result from the model is " +
      //     result.dataSync()
      // );

    // const loadedModel = await tf.loadLayersModel(savePath + "/model.json");
    // const result = loadedModel.predict(tf.tensor2d([[1, 0, 0.5]]));
    // console.log(
    //   "The actual value is 1, the inference result from the model is " +
    //     result.dataSync()
    // );
  } catch (error) {
    console.log("error :>> ", error);
  }
};

run(1000, 1, "file://trainedModel");
