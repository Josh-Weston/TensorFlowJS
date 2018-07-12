let x1 = tf.linspace(-1, 0, 100).add(tf.randomNormal([100], stdev=0.01));
let x2 = tf.linspace(4, 5, 100).add(tf.randomNormal([100], stdev=0.01));

let y1 = tf.linspace(-1, 0, 100).add(tf.randomNormal([100], stdev=0.01));
let y2 = tf.linspace(4, 5, 100).add(tf.randomNormal([100], stdev=0.01));

let xs = x1.concat(x2);
let ys = y1.concat(y2);

let labels = tf.ones([100]).concat(tf.zeros([100]));

tf.plot.scatterColor(xs, ys, '#vis_original', labels);

// We use sigmoid to determine the probability of belonging to each class
let class1 = x1.stack(y1, axis=1),
    class2 = x2.stack(y2, axis=1),
    classes = class1.concat(class2);

let W = tf.randomNormal([2, 1]).variable();
let b = tf.randomNormal([1]).variable();

console.log('initial weights');
W.print();
console.log('initial biases');
b.print();

// let initial_preds = tf.squeeze(tf.matMul(classes, W).add(b));
// inital_preds = tf.round(tf.sigmoid(initial_preds));


//let y_logit = tf.squeeze(tf.matMul(classes, W).add(b)); //Changes from a [2,1] to a [1], the data is the same
let computeLogit = () => {
    let squeezed = tf.squeeze(tf.matMul(classes, W).add(b));  
    return squeezed;  
    //return tf.round(tf.sigmoid(squeezed));
};


const learningRate = 0.20;
const epochs = 100;
let optimizer = tf.train.adam(learningRate);

async function train() {
    for (var i = 0; i < epochs; i++) {
        
        // There just needs to be variables somewhere in the minimize function
        optimizer.minimize(() => {  
            //We pass in logits because the function can handle outliers easily.
            let y_logit = computeLogit();
            let entropy = tf.sigmoidCrossEntropyWithLogits(labels=labels, logits=y_logit); //entropy gets smaller
            let loss = tf.sum(entropy);
            return loss;
        });

        await updatePlot();
    }

    // let preds = tf.squeeze(tf.matMul(classes, W).add(b));
    // let entropy = tf.sigmoidCrossEntropyWithLogits(labels=labels, logits=preds);
    // let loss = tf.sum(entropy);
    // let classPredictions = tf.round(tf.sigmoid(preds));



    // tf.plot.scatterColor(xs, ys, '#vis', classPredictions);

    console.log('Final weights, biases, and loss');
    W.print();
    b.print();
    // loss.print();  
    
    
    // let preds = tf.squeeze(tf.matMul(classes, W).add(b));
    // let classPredictions = tf.round(tf.sigmoid(preds));

    // let slope = -1.4123187,
    //     intercept = 3.2828248;

    // let y_line = xs.mul(slope).add(intercept);
    // tf.plot.fittedScatterColor(xs, ys, xs, y_line, '#vis', classPredictions);

}

function calculateSlope(weights, bias) {
    weights = weights.dataSync();
    bias = bias.dataSync()[0];
    let slope = -(bias/weights[1])/(bias/weights[0]),
        intercept = -(bias/weights[1]);

    return {slope: slope, intercept: intercept};
}

async function updatePlot() {

    let preds = tf.squeeze(tf.matMul(classes, W).add(b));
    let classPredictions = tf.round(tf.sigmoid(preds));
    //await tf.plot.scatterColor(xs, ys, '#vis', classPredictions);

    let {slope, intercept} = calculateSlope(W, b);
    let y_line = xs.mul(slope).add(intercept);
    await tf.plot.fittedScatterColor(xs, ys, xs, y_line, '#vis', classPredictions);

    // Give the browser time to update
    let promise = new Promise((resolve, reject) => {
        setTimeout(() => {
            resolve();
        }, 50);
    });

    await promise;
}  

train();

// Use sigmoid for binary. Sigmoid doesn't really return probabilities, but it converges to 0 and 1 very quickly (e.g. 10 = 1, 0 = 0.5, -10, =0);
// Using sigmoid, we can have a linear function that updates our weights to move the points closer to their
// actual labels based on the cumulative loss from cross-entropy.

// Use softmax for multiclass

// I can't seem to get cross entropy to work

// Softmax: takes a tensor and returns the normalized probabilities. Basically used 
// to pull-out the best "one". tf.softmax(tensor1d[1, 2, 3, 4, 5]) = [0.0116562, 0.0316849, 0.0861286, 0.2341217, 0.6364087]
// Notice how it pushes 5 to the top.

// Logits: means it is performing softmax on the unscaled. The values are not probabilities!
