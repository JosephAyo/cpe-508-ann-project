const tf = require("@tensorflow/tfjs-node");

async function createDataset(csvPath) {
  const dataset = tf.data.csv(csvPath, {
    hasHeader: true,
    columnConfigs: { output: { isLabel: true } },
  });
  const numOfColumns = (await dataset.columnNames()).length - 1;
  // Convert features and labels.
  const createdDataset = {
    dataset: dataset.map((row) => {
      const rawFeatures = row["xs"];
      const rawLabel = row["ys"];
      const convertedFeatures = Object.keys(rawFeatures).map((key) => {
        return rawFeatures[key] / 100;
        // switch (rawFeatures[key]) {
        //   case "A":
        //     return 1;
        //   case "B":
        //     return 2;
        //   case "C":
        //     return 3;
        //   case "D":
        //     return 4;
        //   case "F":
        //     return 5;
        //   default:
        //     return Number(rawFeatures[key]);
        // }
      });
      const convertedLabel = [rawLabel["output"]/100];
      return { xs: convertedFeatures, ys: convertedLabel };
    }),
    numOfColumns: numOfColumns,
  };
  return createdDataset;
}

module.exports = createDataset;
