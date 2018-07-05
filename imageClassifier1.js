// This needs to be async to give the server a chance to load the image data
async function load() {
    let rectImg = new Image(16 ,16),
        circleImg = new Image(16, 16),
        triangleImg = new Image(16, 16),
        ellipseImg = new Image(16, 16);
    
    let img1Promise = new Promise((resolve, reject) => {
        rectImg.src = "./img/rect.png";
        rectImg.onload = resolve; //This sucks, but is necessary
    });

    let img2Promise = new Promise((resolve, reject) => {
        circleImg.src = "./img/circle.png";
        circleImg.onload = resolve; //This sucks, but is necessary
    });

    let img3Promise = new Promise((resolve, reject) => {
        triangleImg.src = "./img/triangle.png";
        triangleImg.onload = resolve; //This sucks, but is necessary
    });

    let img4Promise = new Promise((resolve, reject) => {
        ellipseImg.src = "./img/ellipse.png";
        ellipseImg.onload = resolve; //This sucks, but is necessary
    });


    return Promise.all([img1Promise, img2Promise, img3Promise, img4Promise]).then(() => {
        return {rect: rectImg, circle: circleImg, triangle: triangleImg, ellipse: ellipseImg} //Automatically resolves promise and sends back the data.
    });

}

let tfRect, tfCircle, tfTriangle, tfEllipse;

load().then((images) => {
    // Returns a 3D tensor
    // Each entry is a 2d array of pixels for a single row
    // 1 entry = rgb for 16 pixels for a single row
    let {rect, circle, triangle, ellipse} = images;
    tfRect = tf.fromPixels(rect, 1); //only pull one channel
    tfCircle = tf.fromPixels(circle, 1); //only pull one channel
    tfTriangle = tf.fromPixels(triangle, 1); 
    tfEllipse = tf.fromPixels(ellipse, 1);
    const model = tf.sequential();

    model.add(tf.layers.conv2d({
        inputShape: [16, 16, 1], //Notice how we don't include the batch_size here. They can accept batches of arbitrary size.
        kernelSize: 4, //4x4 filter window
        filters: 8, //number of windows of size kernelSize to apply to the image
        strides: 1, //step size of the sliding window. Move 1 pixel at a time
        activation: 'relu',
        kernelInitialzer: 'VarianceScaling'
    }));

    model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2], //Apply a 2x2 window to the pooling data
        strides: [2, 2] //the filter will slide over the image in steps of 2 pixels in both horiztonal and veritcal directions
    }));

    //The poolSize and strides are non-overlapping. This means the pooling layer wil cut the size of the activation from
    //the previous layer in half.

    // Add a couple more layers. Here we don't specify an input shape because it will be inferred from the previous layer
    model.add(tf.layers.conv2d({
        kernelSize: 4,
        filters: 16,
        strides: 1,
        //activation: 'relu',
        activation: 'tanh', //Changing to tanh solved my problem! But I don't know why
        kernelInitializer: 'VarianceScaling'
      }));
      
    model.add(tf.layers.maxPooling2d({
        poolSize: [2, 2],
        strides: [2, 2]
    }));

    //Flattens the ouput
    model.add(tf.layers.flatten());

    //Add a dense layer for the actual classification
    //2 units since we only have 2 possibilities. Might be able to use only 1?
    model.add(tf.layers.dense({
        units: 4,
        kernelInitializer: 'VarianceScaling',
        // kernelInitializer: 'randomNormal',
        activation: 'softmax' //Creates a probability for each class
    }));

    /* Training the model */
    const LEARNING_RATE = 0.15;
    const optimizer = tf.train.sgd(LEARNING_RATE);

    // We use categorialCrossEntropy to calculate loss of the labels
    model.compile({
        optimizer: optimizer,
        loss: 'categoricalCrossentropy',
        metrics: ['accuracy']
    });

    // How many examples the model should "see" before updating parameters
    const BATCH_SIZE = 30; //[10, 16, 16, 16] is the actual shape of our data with batching

    //How many batches to train in total
    const TRAIN_BATCHES = 60;

/*     As discussed earlier, the dimensionality of a single image in our MNIST data set is [28, 28, 1]. 
    When we set a BATCH_SIZE of 64, we're batching 64 images at a time, which means the actual shape of our
    data is [64, 28, 28, 1] (the batch is always the outermost dimension). */

    function createData(BATCH_SIZE) {
        
        let batch = [];
        let ysArray = [];
        for (var i = 0; i < BATCH_SIZE; i++) {
            let random = Math.random();
            switch (true) {
                case random < .25:
                    batch.push(tfCircle.clone());
                    ysArray.push([1, 0, 0, 0]);
                    break;
                case random < .50:
                    batch.push(tfRect.clone());
                    ysArray.push([0, 1, 0, 0]);
                    break;
                case random < .70:
                    batch.push(tfTriangle.clone());
                    ysArray.push([0, 0, 1, 0]);
                    break;
                default:
                    batch.push(tfEllipse.clone());
                    ysArray.push([0, 0, 0, 1]);
                    break;
            }
        }

        xsArray = tf.stack(batch).toFloat();
        ysArray = tf.tensor2d(ysArray); //const labels = tf.tensor2d(batchLabelsArray, [batchSize, NUM_CLASSES]);
        return {xs: xsArray, ys: ysArray};
    }

    function createTestData(BATCH_SIZE) {
        
        let batch = [];
        let ysArray = [];
        for (var i = 0; i < BATCH_SIZE; i++) {

            let random = Math.random();
            switch (true) {
                case random < .25:
                    batch.push(tfCircle.clone());
                    ysArray.push([1, 0, 0, 0]);
                    break;
                case random < .50:
                    batch.push(tfRect.clone());
                    ysArray.push([0, 1, 0, 0]);
                    break;
                case random < .70:
                    batch.push(tfTriangle.clone());
                    ysArray.push([0, 0, 1, 0]);
                    break;
                default:
                    batch.push(tfEllipse.clone());
                    ysArray.push([0, 0, 0, 1]);
                    break;
            }

        }

        xsArray = tf.stack(batch).toFloat();
        ysArray = tf.tensor2d(ysArray); //const labels = tf.tensor2d(batchLabelsArray, [batchSize, NUM_CLASSES]);
        return {xs: xsArray, ys: ysArray};
    }


    //tfRect.expandDims(); adds another dimension to the front
    //tfRect.reshape([1, 16, 16, 1]); adds another dimension to the front
    //tfRect.stack(tfRect.clone()); adds another dimension, but allows us to stack them. Really only works to add the dimension on the first stack. After than, you need the proper dimensions.
    //tfRect4d.concat(tfRect.clone()); concatenates if dimension already exists

    async function fitModel() {
        //Train the model
        for (let i = 0; i < TRAIN_BATCHES; i++) {

            let validationData;
            if (i % 2 === 0) {
                let {xs: testXs, ys: testYs} = createTestData(BATCH_SIZE);
                validation = [testXs, testYs];
            }

            let {xs, ys} = createData(BATCH_SIZE);
            const history = await model.fit(
                xs,
                ys,
                {
                    batchSize: BATCH_SIZE,
                    validationData, //What the model is evaluated against
                    epochs: 1
                });

            const loss = history.history.loss[0];
            const accuracy = history.history.acc[0];
            console.log(loss, accuracy);

        }

    } 

    fitModel().then(() => {
        
        // The output is the class probabilities or each
        let predictions = model.predict(tfRect.clone().expandDims().toFloat()).dataSync();
        checkPredictions('rect', predictions);
        predictions = model.predict(tfTriangle.clone().expandDims().toFloat()).dataSync();
        checkPredictions('triangle', predictions);
        predictions = model.predict(tfEllipse.clone().expandDims().toFloat()).dataSync();
        checkPredictions('ellipse', predictions);
        predictions = model.predict(tfCircle.clone().expandDims().toFloat()).dataSync();
        checkPredictions('circle', predictions);
        predictions = model.predict(tfTriangle.clone().expandDims().toFloat()).dataSync();
        checkPredictions('triangle', predictions);
        predictions = model.predict(tfRect.clone().expandDims().toFloat()).dataSync();
        checkPredictions('rect', predictions);
        console.log('model is done')
    });

    // Pretty good. It gets them all right.
    function checkPredictions(actual, predictions) {
        let position = ['circle', 'rect', 'triangle', 'ellipse'];
        let index = predictions.indexOf(predictions.reduce((largest, current) => {
            return Math.max(largest, current);
        }, 0));
        console.log(`Actual: ${actual}, Prediction: ${position[index]}`);
    }

});


// TODO:
//http://cv-tricks.com/tensorflow-tutorial/training-convolutional-neural-network-for-image-classification/
//Color images as well