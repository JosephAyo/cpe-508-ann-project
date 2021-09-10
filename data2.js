const tf = require("@tensorflow/tfjs-node");

async function createDataset(csvPath) {
  const dataset = tf.data.csv(csvPath, {
    hasHeader: true,
    columnConfigs: { output: { isLabel: true } },
  });
  const numOfColumns = (await dataset.columnNames()).length - 1;
  // Convert features and labels.
  return {
    dataset: dataset.map((row) => {
      const rawFeatures = row["xs"];
      const rawLabel = row["ys"];
      const convertedFeatures = Object.keys(rawFeatures).map((key) => {
        switch (rawFeatures[key]) {
          case "A":
            return 1;
          case "B":
            return 2;
          case "C":
            return 3;
          case "D":
            return 4;
          case "F":
            return 5;
          default:
            return Number(rawFeatures[key]);
        }
      });
      const convertedLabel = [rawLabel["output"]];
      return { xs: convertedFeatures, ys: convertedLabel };
    }),
    numOfColumns: numOfColumns,
  };
}

module.exports = createDataset;
