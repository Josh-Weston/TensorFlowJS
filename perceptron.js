// Problem given a specific fruit, can we predict what type it is based on: [shape, texture, weight]
//We have two inputs, a single neuron, and a binary output

function createData() {

    let xsArray = [];
    let ysArray = [];
    for (var i = 0; i < 50; i++) {
        if (i % 2 == 0) {
            xsArray.push([1, 1, -1]) //Apples on even
            ysArray.push(1);
        } else {
            xsArray.push([1, -1, -1]); //Orages on odd
            ysArray.push(-1);
        }
    }

    return {xs: xsArray, ys: ysArray}

}

let {xs, ys} = createData();
xs = tf.tensor2d(xs); //Training data
ys = tf.tensor1d(ys); //Labels

function myActivation(input) {
    console.log(input);
}

let model = tf.sequential({
    layers: [tf.layers.dense({units: 1, useBias: true, inputShape: [3], activation: 'tanh'})] //Using an activation function seems to generalize better though
    //layers: [tf.layers.dense({units: 1, useBias: true, inputShape: [3]})] //Model performs better without an activation function when overfitting
});


model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});


async function fitModel() {
    for (let i = 1; i < 10; ++i) {
        let h = await model.fit(xs, ys, {
            batchSize: 4,
            epochs: 3
        });
        console.log("Loss after Epoch " + i + " : " + h.history.loss[0]);
    }

    return new Promise((resolve, reject) => resolve());
}

fitModel().then(() => {

    let result = model.evaluate(
        xs, ys);
    
    console.log('Error: ');
    result.print();

    predictions = model.predict(tf.tensor2d([[1, 1, -1], [1, 1, -1], [1, -1, -1], [1, 0, -1]])).dataSync(); //0 in the middle depends on the model's weights after training
    //Get the data
    predictions.forEach(pred => {
        if (pred > 0) {
            console.log(`${pred}: Apple`);
        } else {
            console.log(`${pred}: Orange`);
        }
    });

});



// let result = model.evaluate(
//     xs, ys);
// result.print();

// model.predict(tf.tensor2d([[1, 1, -1], [1, 1, -1], [1, -1, -1]])).print();



/* //Create the model
let model = tf.model({
    inputs: inputShape,
    outputs: [denseOutput, activationOutput]
});

//Collect both outputs and print separately
let [denseOut, activationOut] = model.predict(tf.tensor2d([[2.0]]))
denseOut.print();
activationOut.print(); */


/* const model = tf.sequential({
    layers: [tf.layers.dense({units: 1, inputShape: [10]})]
 });
 model.compile({optimizer: 'sgd', loss: 'meanSquaredError'});
 const result = model.evaluate(
      tf.ones([8, 10]), tf.ones([8, 1]), {batchSize: 4});
 result.print(); */












// This is a basic predictive model, but it has nothing to do with neural networks!!
// It will change the variables as required to minimize loss

/* // Fit a quadratic function by learning the coefficients a, b, c.
const xs = tf.tensor1d([0, 1, 2, 3]);
const ys = tf.tensor1d([1.1, 5.9, 16.8, 33.9]);

const a = tf.scalar(Math.random()).variable();
const b = tf.scalar(Math.random()).variable();
const c = tf.scalar(Math.random()).variable();

// y = a * x^2 + b * x + c.
const f = x => a.mul(x.square()).add(b.mul(x)).add(c);
const loss = (pred, label) => pred.sub(label).square().mean();

const learningRate = 0.01;
const optimizer = tf.train.sgd(learningRate);

// Train the model.
for (let i = 0; i < 10; i++) {
   optimizer.minimize(() => loss(f(xs), ys));
}

// Make predictions.
console.log(
     `a: ${a.dataSync()}, b: ${b.dataSync()}, c: ${c.dataSync()}`);
const preds = f(xs).dataSync();
preds.forEach((pred, i) => {
   console.log(`x: ${i}, pred: ${pred}`);
}); */


