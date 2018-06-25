'use strict'

/* // Define a model for linear regression.
let model = tf.sequential();
model.add(tf.layers.dense({units: 1, inputShape: [1]}));

// Prepare the model for training: Specify the loss and the optimizer.
model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

// Generate some synthetic data for training.
const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

// Train the model using the data.
model.fit(xs, ys, {epochs: 10}).then(() => {
// Use the model to do inference on a data point the model hasn't seen before:
model.predict(tf.tensor2d([5], [1, 1])).print();
});



let activationLayer = tf.layers.activation({activation: 'sigmoid'}); */



// denseLayer = tf.layers.dense({
//     units: 1,
//     kernelInitializer: 'zeros',
//     useBias: false
// });

// input = tf.ones([2,2]);
// output = denseLayer.apply(input);

// output.print();

let denseLayer = tf.layers.dense({
    units: 2,
    // kernelInitializer: 'ones',
    //kernelInitializer: tf.initializers.constant({value: 3}), //This gives me a single constant
    useBias: true
 });

// Invoke the layer's apply() method with a [tf.Tensor](#class:Tensor) (with concrete
// numeric values).
// let input = tf.ones([2, 2]);
// Set the initial shape for the activation layer

// YES! This works!

let inputShape = tf.input({shape: [2, 3]});
denseLayer.apply(inputShape);

// We can only apply our weights after data has been applied
let weightsVariable = tf.variable(tf.tensor2d([[500, 200], [200, 200], [300, 200]]), true);
let biasVariable = tf.variable(tf.tensor([0, 0]), true);
denseLayer.setWeights([weightsVariable, biasVariable]);
let input = tf.tensor2d([[1, 2, 3], [4, 5, 6]]); //It is always expecting a 2D array. This would be one set of possible inputs
let output = denseLayer.apply(input);

// The output's value is expected to be [[0], [0]], due to the fact that
// the dense layer has a kernel initialized to all-zeros and does not have
// a bias.
input.print(); //Input
output.print(); //Output

//Denselayer weights:
console.log('Dense Layer weights');
denseLayer.getWeights()[0].print(); //The actual weights
denseLayer.getWeights()[1].print(); //The bias

console.log('#### PROBLEM 1 ####');
// input = 2.0, weight = 2.3, bias is -3
denseLayer = tf.layers.dense({
    units: 1,
    useBias: true
});

inputShape = tf.input({shape: [1, 1]});
denseLayer.apply(inputShape);

weightsVariable = tf.variable(tf.tensor2d([[2.3]]), true);
biasVariable = tf.variable(tf.tensor1d([-3]), true);
denseLayer.setWeights([weightsVariable, biasVariable]);
input = tf.tensor2d([[2.0]]);
output = denseLayer.apply(input);

input.print();
output.print();

console.log('#### PROBLEM 2 ####');
// input = 2.0, weight = 2.3, bias is -3
// input = 2.0, weight = 2.3, bias is -3
denseLayer = tf.layers.dense({
    units: 1,
    useBias: true,
    activation: 'sigmoid'
});

inputShape = tf.input({shape: [1, 1]});
denseLayer.apply(inputShape);

weightsVariable = tf.variable(tf.tensor2d([[2.3]]), true);
biasVariable = tf.variable(tf.tensor1d([-3]), true);
denseLayer.setWeights([weightsVariable, biasVariable]);
input = tf.tensor2d([[2.0]]);
output = denseLayer.apply(input);
output.print();



// A FULL MODEL, with dense layer and activation layer separated
inputShape = tf.input({shape: [1]}) //sames as [x, 1]
denseLayer = tf.layers.dense({units: 1, useBias: true});
let activationLayer = tf.layers.activation({activation: 'sigmoid'});

let denseOutput = denseLayer.apply(inputShape);

// *** purposefully set the weigths and biases ***
weightsVariable = tf.variable(tf.tensor2d([[2.3]]), true);
biasVariable = tf.variable(tf.tensor1d([-3]), true);
denseLayer.setWeights([weightsVariable, biasVariable]);
// *** purposefully set the weigths and biases ***

let activationOutput = activationLayer.apply(denseOutput);

//Create the model
let model = tf.model({
    inputs: inputShape,
    outputs: [denseOutput, activationOutput]
});

//Collect both outputs and print separately
let [denseOut, activationOut] = model.predict(tf.tensor2d([[2.0]]))
denseOut.print();
activationOut.print();


//PROBLEM 3
//2 input neuron, b = 1.2, w = [3, 2], p = [-5, 6]
// Activation layer as part of denslayer
inputShape = tf.input({shape: [2]}) //sames as [x, 1]
denseLayer = tf.layers.dense({units: 2, useBias: true, activation: 'linear'});
denseLayer.apply(inputShape);

// *** purposefully set the weigths and biases ***
weightsVariable = tf.variable(tf.tensor2d([3, 2, 4, 6], [2,2]), true); //Use the shape parameter for legibility. This creates 2 columns. 1 column for each neuron
biasVariable = tf.variable(tf.tensor1d([0, 0]), true);
denseLayer.setWeights([weightsVariable, biasVariable]);
input = tf.tensor2d([[1, 1], [2, 2]]);
output = denseLayer.apply(input);
output.print();

/* 
    Units in denselayer = dimensionality of the output space, in other words, number of neurons.
    We need a weights matrix for each unit in the denselayer

*/