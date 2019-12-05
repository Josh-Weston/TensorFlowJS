var ws = new WebSocket("ws://127.0.0.1:9999/");

let values;
let messagesQueue = [];
let started = false;

ws.onmessage = function (event) {
    
    msgs = JSON.parse(event.data);
    switch (msgs['type']) {
        case 'INIT':
            init(msgs);
            break;
        case 'STEP':
            messagesQueue.push(msgs);
            if (started === false) {
                started = true;
                step();
            }
            break;
        default:
            console.log('WOAH I got something weird: ' + msgs['type']);
    }

}

async function init(data){
    total_steps = data['num_steps']
    document.getElementById('top').innerHTML = 'Real Slope = ' + data['true_w'] + ', Real Y intercept = ' + data['true_b']

    values = data.xs.map((el, i) => {
        return {x: el, y: data.ys[i]}
    });

    let spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v2.json",
        "width": 600,
        "height": 600,
        "description": "A scatterplot showing progress of ML model.",
        "data": {"values" : values},
        "mark": "point",
        "encoding": {
            "x": {"field": "x","type": "quantitative"},
            "y": {"field": "y","type": "quantitative"}
        }
    }

    // Give the browser time to update
    let promise = new Promise((resolve, reject) => {
        setTimeout(() => {
            resolve();
        }, 5);
    });

    await vegaEmbed('#vis', spec);

}

async function step(){

    while (messagesQueue.length > 0) {
        var msg = messagesQueue.shift();
        document.getElementById('progress').innerHTML = 'Step ' + msg['num'] + ' / ' + total_steps + ', Loss = ' + msg['loss']
        document.getElementById('bottom').innerHTML = 'Predicted Slope = ' + msg['w'] + ', Predicted Y Intercept = ' + msg['b']
        let points = [];
        points.push({x: 0, y: msg.b, color: 'red'});
        points.push({x: 1, y: 1*msg.w + msg.b, color:'red'});
        
        await drawChart(points);
    
        // Give the browser time to update
        let promise = new Promise((resolve, reject) => {
            setTimeout(() => {
                resolve();
            }, 5);
        });
    
        await promise;
    } 
    document.getElementById('progress').innerHTML = 'Completed ' + total_steps + ' Steps, Loss = ' + msg['loss']

    console.log('Messges queue is empty');
}

async function drawChart(points) {

    let spec = {
        "$schema": "https://vega.github.io/schema/vega-lite/v2.json",
        "width": 600,
        "height": 600,
        "description": "A scatterplot showing progress of ML model.",
        'layer': [
            {
                'data': {'values': values},
                'mark': 'point',
                'encoding': {
                'x': {'field': 'x', 'type': 'quantitative'},
                'y': {'field': 'y', 'type': 'quantitative'},
                'color': {'field': 'color', 'type': 'nominal'}
                }
            },
            {
                'data': {'values': points},
                'mark': 'line',
                'encoding': {
                'x': {'field': 'x', 'type': 'quantitative'},
                'y': {'field': 'y', 'type': 'quantitative'},
                'color': {'field': 'color', 'type': 'nominal'}
                }
            }
        ]
    };

    await vegaEmbed('#vis', spec);

}
