import * as math from "mathjs";

// Compute cost for linear regression
// - computeCost: computes the cost of using theta as the
//      parameter for linear regression to fit the data points in X and y
export function computeCost(X, y, theta) {
  // number of training examples
  const m = y.length;

  // z = X*theta - y;
  const z = math.subtract(math.multiply(X, theta), y);

  // J = (z'*z)/(2*m)
  const J = math.divide(math.multiply(math.transpose(z), z), 2 * m);

  return J;
}
