var yaxtor = yaxtor || {};
(function(){
    var _this = this

    this.init = function() {
        var dsv  = require('d3-dsv'),
            fs   = require('fs'),
            math = require('mathjs')

        var data = dsv.csvParseRows(fs.readFileSync("./data/ex1data1.txt", "utf-8"))
        var m = data.length
        // console.log(m)

        var X = math.subset(data, math.index(math.range(0, m), 0))
        var y = math.subset(data, math.index(math.range(0, m), 1))
        // console.log('X: ', X)
        // console.log('y: ', y)

        // Gradient Descent...
        console.log('Running Gradient Descent ...\n')

        // Add a column of ones to x
        X = math.concat(math.ones([m, 1]), X)
        // console.log(X)

        // initialize fitting parameters
        var theta =  math.zeros([2, 1])
        // console.log(theta) 

        // Some gradient descent settings
        var iterations = 1000;
        var alpha = 0.01;

        // compute and display initial cost
        var computeCost = require('./src/computeCost')
        J = computeCost(X, y, theta)
        console.log('Initial cost: ', J)

        // run gradient descent
        var gradientDescent = require('./src/gradientDescent')
        var g_result = gradientDescent(X, y, theta, alpha, iterations)
        console.log('Last cost: ', g_result['J_history'][iterations-1])
    }
}).apply(yaxtor);

// Run
yaxtor.init();