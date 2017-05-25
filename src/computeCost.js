var math = require('mathjs')

// Compute cost for linear regression
// - computeCost: computes the cost of using theta as the
//      parameter for linear regression to fit the data points in X and y
var computeCost = function (X, y, theta) {

    // Initialize some useful values
    m = y.length // number of training examples

    J = 0

    // z = X*theta - y;
    z = math.subtract(math.multiply(X, theta), y) 

    // J = (z'*z)/(2*m)
    J = math.divide(math.multiply(math.transpose(z), z), 2*m) 
    return J
}

module.exports = computeCost;