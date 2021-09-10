const tf = require("@tensorflow/tfjs-node");

async function createDataset(csvPath) {
  const dataset = tf.data.csv(csvPath, {
    hasHeader: true,
    columnConfigs: { rings: { isLabel: true } },
  });
  const numOfColumns = (await dataset.columnNames()).length - 1;
  // Convert features and labels.
  return {
    dataset: dataset.map((row) => {
      const rawFeatures = row["xs"];
      const rawLabel = row["ys"];
      const convertedFeatures = Object.keys(rawFeatures).map((key) => {
        switch (rawFeatures[key]) {
          case "F":
            return 0;
          case "M":
            return 1;
          case "I":
            return 2;
          default:
            return Number(rawFeatures[key]);
        }
      });
      const convertedLabel = [rawLabel["rings"]];
      return { xs: convertedFeatures, ys: convertedLabel };
    }),
    numOfColumns: numOfColumns,
  };
}

module.exports = createDataset;
