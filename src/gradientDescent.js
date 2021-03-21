import * as math from "mathjs";
import { computeCost } from "./computeCost.js";

// Performs gradient descent to learn theta
// - gradientDescent: updates theta by taking num_iters gradient
//      steps with learning rate alpha
export function gradientDescent(X, y, theta, alpha, num_iters) {
  // number of training examples
  const m = y.length;

  // initialize J history
  let J_history = math.zeros([num_iters, 1]);

  for (let iter = 0; iter < num_iters; iter++) {
    // theta = theta - (alpha*(X'*(X*theta - y))/m)
    const dt = math.multiply(
      math.transpose(X),
      math.subtract(math.multiply(X, theta), y)
    );

    theta = math.subtract(theta, math.divide(math.multiply(alpha, dt), m));

    // Save the cost J in every iteration
    J_history[iter] = computeCost(X, y, theta);
  }

  return { theta, J_history };
}
