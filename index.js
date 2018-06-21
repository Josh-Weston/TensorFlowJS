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

const denseLayer = tf.layers.dense({
    units: 1,
    // kernelInitializer: 'ones',
    kernelInitializer: tf.initializers.constant({value: tf.tensor([1, 2, 3], [1, 3])}), //This gives me a single constant
    useBias: true
 });
 
// Invoke the layer's apply() method with a [tf.Tensor](#class:Tensor) (with concrete
// numeric values).
// let input = tf.ones([2, 2]);
input = tf.tensor2d([[1, 2, 3]]); //It is always expecting a 2D array. This would be one set of possible inputs
const output = denseLayer.apply(input);

// The output's value is expected to be [[0], [0]], due to the fact that
// the dense layer has a kernel initialized to all-zeros and does not have
// a bias.
input.print(); //Input
output.print(); //Output


//Denselayer weights:
console.log('Dense Layer weights');
denseLayer.getWeights()[0].print(); //The actual weights
denseLayer.getWeights()[1].print(); //The bias

console.log(tf.initializers.Initializer)