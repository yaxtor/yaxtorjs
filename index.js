import dsv from "d3-dsv";
import { promises as fsp } from "fs";
import * as math from "mathjs";

import { computeCost } from "./src/computeCost.js";
import { gradientDescent } from "./src/gradientDescent.js";

class Yaxtor {
  static async run(options) {
    try {
      const {
        path,
        encoding = "utf-8",
        // Some gradient descent settings
        iterations = 1000,
        alpha = 0.01,
      } = options;

      // read data
      const file = await fsp.readFile(path, encoding);
      const data = dsv.csvParseRows(file);

      const m = data.length;

      let X = math.subset(data, math.index(math.range(0, m), 0));
      const y = math.subset(data, math.index(math.range(0, m), 1));

      // Gradient Descent...
      console.log("Running Gradient Descent ...\n");

      // Add a column of ones to x
      X = math.concat(math.ones([m, 1]), X);

      // initialize fitting parameters
      const theta = math.zeros([2, 1]);

      // compute and display initial cost
      const J = computeCost(X, y, theta);

      console.log("Initial cost: ", J);
      console.log("Theta: ", theta, "\n");

      // run gradient descent
      const result = gradientDescent(X, y, theta, alpha, iterations);

      console.log("Last cost: ", result.J_history[iterations - 1]);
      console.log("Theta: ", result.theta, "\n");
    } catch (error) {
      console.error(error);
    }
  }
}

// Run
Yaxtor.run({ path: "./data/ex1data1.txt" });
