// Creating toy datasets
let n = 100,
    wTrue = 5, //trye weight
    bTrue = 2, //true bias
    noiseScale = .1;

// Create a linear function with a bit of randomization
let x = tf.randomUniform([n], minval=0, maxval=1);
let noise = tf.randomNormal([n], mean=0, stdev=noiseScale);
let y = x.mul(tf.scalar(wTrue)).add(tf.scalar(bTrue)).add(noise); //Actual labels

const wVar = tf.scalar(Math.random()).variable();
const bVar = tf.scalar(Math.random()).variable();

const linearFunction = x => x.mul(wVar).add(bVar);

tf.plot.scatter(x, y, '#vis');

const learningRate = 0.1;
const optimizer = tf.train.sgd(learningRate);

async function train() {
    //Train the model
    for (let i = 0; i < 400; i++) {
        optimizer.minimize(() => tf.losses.meanSquaredError(linearFunction(x), y));
        await updatePlot();
    }
}

async function updatePlot() {
    const preds = linearFunction(x);
    await tf.plot.fittedScatter(x, y, x, preds, '#vis');

    // Give the browser time to update
    let promise = new Promise((resolve, reject) => {
        setTimeout(() => {
            resolve();
        }, 5);
    });

    await promise;

}   

// This doesn't change as we make client side changes
// Only during loading does it do anything!
document.addEventListener('readystatechange', event => {
    console.log(event.target.readyState);
});

train();