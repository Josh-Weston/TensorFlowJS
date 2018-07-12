class Plot {
    
    constructor(vegaEmbed) {
        this.vegaEmbed = vegaEmbed;
    }

    scatter(xTensor, yTensor, element) {

        // Vega embed code
        const xVals = xTensor.dataSync();
        const yVals = yTensor.dataSync();

        // Do this so we can feed in JSON array
        const values = Array.from(yVals).map((y, i) => {
            return {'x': xVals[i], 'y': yVals[i]};
        });

        const spec = {
            '$schema': 'https://vega.github.io/schema/vega-lite/v2.json',
            'width': 500,
            'height': 500,
            'selection': {
                'grid': {
                    'type': 'interval', 'bind': 'scales'
                }
            },
            'data': {'values': values},
            'mark': 'point',
            'encoding': {
            'x': {'field': 'x', 'type': 'quantitative'},
            'y': {'field': 'y', 'type': 'quantitative'}
            }
        };
        this.vegaEmbed(element, spec);
    }

    async scatterColor(xTensor, yTensor, element, color) {
        // Vega embed code
        const xVals = xTensor.dataSync();
        const yVals = yTensor.dataSync();
        const colorVals = color.dataSync();

        // Do this so we can feed in JSON array
        const values = Array.from(yVals).map((y, i) => {
            return {'x': xVals[i], 'y': yVals[i], 'color': colorVals[i]};
        });

        const spec = {
            '$schema': 'https://vega.github.io/schema/vega-lite/v2.json',
            'width': 500,
            'height': 500,
            'selection': {
                'grid': {
                    'type': 'interval', 'bind': 'scales'
                }
            },
            'data': {'values': values},
            'mark': 'point',
            'encoding': {
            'x': {'field': 'x', 'type': 'quantitative'},
            'y': {'field': 'y', 'type': 'quantitative'},
            'color': {'field': 'color', 'type': 'nominal'}
            }
        };
        await this.vegaEmbed(element, spec);
    }

    async fittedScatterColor(xTensor, yTensor, xTensorLine, yTensorLine, element, color) {

        // Vega embed code
        const xVals = xTensor.dataSync();
        const yVals = yTensor.dataSync();
        const xValsLine = xTensorLine.dataSync();
        const yValsLine = yTensorLine.dataSync();
        const colorVals = color.dataSync()

        // Do this so we can feed in JSON array
        const valuesScatter = Array.from(yVals).map((y, i) => {
            return {'x': xVals[i], 'y': yVals[i], 'color': colorVals[i]};
        });

        const valuesLine = Array.from(yValsLine).map((y, i) => {
            return {'x': xValsLine[i], 'y': yValsLine[i]};
        });

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
                        'data': {'values': valuesScatter},
                        'mark': 'point',
                        'encoding': {
                        'x': {'field': 'x', 'type': 'quantitative'},
                        'y': {'field': 'y', 'type': 'quantitative'},
                        'color': {'field': 'color', 'type': 'nominal'}
                        }
                    },
                    {
                        'data': {'values': valuesLine},
                        'mark': 'line',
                        'encoding': {
                        'x': {'field': 'x', 'type': 'quantitative'},
                        'y': {'field': 'y', 'type': 'quantitative'}
                        }
                    }
                ]
        };

        await this.vegaEmbed(element, spec);
    }
    
    async fittedScatter(xTensor, yTensor, xTensorLine, yTensorLine, element) {

        // Vega embed code
        const xVals = xTensor.dataSync();
        const yVals = yTensor.dataSync();
        const xValsLine = xTensorLine.dataSync();
        const yValsLine = yTensorLine.dataSync();

        // Do this so we can feed in JSON array
        const valuesScatter = Array.from(yVals).map((y, i) => {
            return {'x': xVals[i], 'y': yVals[i]};
        });

        const valuesLine = Array.from(yValsLine).map((y, i) => {
            return {'x': xValsLine[i], 'y': yValsLine[i]};
        });

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
                        'data': {'values': valuesScatter},
                        'mark': 'point',
                        'encoding': {
                        'x': {'field': 'x', 'type': 'quantitative'},
                        'y': {'field': 'y', 'type': 'quantitative'}
                        }
                    },
                    {
                        'data': {'values': valuesLine},
                        'mark': 'line',
                        'encoding': {
                        'x': {'field': 'x', 'type': 'quantitative'},
                        'y': {'field': 'y', 'type': 'quantitative'}
                        }
                    }
                ]
        };

        await this.vegaEmbed(element, spec);
    }

}

if (tf) {
    tf.plot = new Plot(vegaEmbed);
} else {
    console.log('You do not have tensorflow loaded');
}


