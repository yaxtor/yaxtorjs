var math = require('mathjs'),
    computeCost = require('./computeCost')

// Performs gradient descent to learn theta
// - gradientDescent: updates theta by taking num_iters gradient 
//      steps with learning rate alpha
var gradientDescent = function(X, y, theta, alpha, num_iters) {
    
    // Initialize some useful values
    m = y.length // number of training examples
    
    J_history = math.zeros([num_iters, 1])

    for (iter = 0; iter < num_iters; iter++) {
        // theta = theta - (alpha*(X'*(X*theta - y))/m)
        dt = math.multiply(math.transpose(X), math.subtract(math.multiply(X, theta), y))
        // console.log('dt: ', dt)
        theta = math.subtract(theta, math.divide(math.multiply(alpha, dt), m))
        // console.log('theta: ', theta)
        
        // Save the cost J in every iteration    
        J_history[iter] = computeCost(X, y, theta)
        // console.log('iteration: ' + iter, J_history[iter])
    }
    return {
        'theta': theta,
        'J_history': J_history
    }
}

module.exports = gradientDescent