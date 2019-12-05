fetch('./data/sales_item.json')
    .then(res => res.json())
    .catch(error => console.error(error))
    .then(buildModel);

const numEpochs = 5;

async function buildModel(data) {
    
    console.log('here');
    //Build a single column time series, and scale it
    let timeSeries = data.map(el => +el[1]);
    let extent = d3.extent(timeSeries);
    let scale = d3.scaleLinear().domain(extent).range([0, 1]);
    timeSeries = timeSeries.map(el => scale(el));

    //Shift Y forward
    let lookBack = 1;
    let dataX = timeSeries;
    let dataY = timeSeries.slice(1);
    
    //Split into train and test sets.
    let train_size = timeSeries.length * .67;
    let trainX = dataX.slice(0,train_size);
    let trainY = tf.tensor1d(dataY.slice(0, train_size));
    let testX = dataX.slice(train_size - 1);
    let testY = tf.tensor1d(dataY.slice(train_size - 1));

    //Reshape data into 3D array [samples, time steps, features]
    trainX = tf.tensor3d(trainX, [trainX.length, lookBack, 1]);
    testX = tf.tensor3d(testX, [testX.length, lookBack, 1]);

    //Create and fit the LSTM model
    const model = tf.sequential();
    model.add(tf.layers.lstm({
        units: 4,
        inputShape: [1, lookBack],
    }));

    model.add(tf.layers.dense({units:1}));
    model.compile({
        optimizer: 'adam',
        loss: 'meanSquaredError'
        //metrics: ['accuracy']
    });

    const results = await model.fit(trainX, trainY, {
        batchSize: 1,
        epochs: numEpochs
    });

    console.log('fitted');
    console.log(results);

    // Make predictions
    let trainPredict = model.predict(trainX);
    let testPredict = model.predict(testX);

    let trainPredictData = Array.from(trainPredict.dataSync()); // Returns a typed array
    let testPredictData = Array.from(testPredict.dataSync()); // Returns a typed array
    let combinedData = trainPredictData.concat(testPredictData);
    unscaledY = combinedData.map(data => scale.invert(data));
    unscaledX = dataX.map(data => scale.invert(data));

    // This isn't a scatter! X = t0
    let plotData = unscaledX.map((data, i) => {
        if (i < trainPredictData.length) {
            return {x: i, y: unscaledY[i], color: 'train'};
        } else {
            return {x: i, y: unscaledY[i], color: 'test'};
        }
    });

    // let plotOriginalData = unscaledX.map((data, i) => {
    //     return {x: i, original: data, color: 'original'};
    // });

    const spec = {
        '$schema': 'https://vega.github.io/schema/vega-lite/v2.json',
        'width': 500,
        'height': 500,
        'selection': {
            'grid': {
                'type': 'interval', 'bind': 'scales'
            }
        },
        'layer': [
            {
                'data': {'values': plotData},
                'mark': 'line',
                'encoding': {
                    'x': {'field': 'x', 'type': 'quantitative'},
                    'y': {'field': 'y', 'type': 'quantitative'},
                    'color': {'field': 'color', 'type': 'nominal'},
                    "tooltip": [
                        {"field": "x", "type": "quantitative"},
                        {"field": "y", "type": "quantitative"}
                    ]
                }
            }
            // {
            //     'data': {'values': plotOriginalData},
            //     'mark': 'line',
            //     'encoding': {
            //         'x': {'field': 'x', 'type': 'quantitative'},
            //         'y': {'field': 'original', 'type': 'quantitative'},
            //         'color': {'field': 'color', 'type': 'nominal'}
            //     }
            // }

        ]
    };

    await this.vegaEmbed('#vis', spec);
}