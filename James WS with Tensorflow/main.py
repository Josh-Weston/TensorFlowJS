print('Importing Modules')
import asyncio
import websockets
import os.path
import warnings
with warnings.catch_warnings(record = True) as w:
    import tensorflow as tf        
import numpy as np
from json import dumps
print('Done Importing')



def make_data(N=100, w=5, b=2, noise_scale=0.1):
    
    x_np = np.random.rand(N, )
    noise = np.random.normal(scale=noise_scale, size=(N, ))
    y_np = w * x_np + b+noise
    global TRUE_W, TRUE_B
    TRUE_W, TRUE_B = w, b

    return {'x': x_np, 'y': y_np}



async def model(x_np, y_np, ws, rate=0.04):

    tf.reset_default_graph()
    
    with tf.name_scope('placeholders'):
        x = tf.placeholder(tf.float32, (100, ))
        y = tf.placeholder(tf.float32, (100, ))

    with tf.name_scope('weights'):
        w = tf.Variable(tf.random_normal((1,)))
        b = tf.Variable(tf.random_normal((1,)))
        
    with tf.name_scope('prediction'):
        y_pred = w*x + b
        
    with tf.name_scope('loss'):
        l = tf.reduce_sum((y-y_pred)**2)
        
    with tf.name_scope('optim'):
        train_op = tf.train.AdamOptimizer(rate).minimize(l)
        
    with tf.name_scope('summaries'):
        tf.summary.scalar('loss', l)
        merged = tf.summary.merge_all()
        
    train_writer = tf.summary.FileWriter('../lr-train', tf.get_default_graph())

    n_steps = 1000

    mess = {'type': 'INIT',
            'true_w': TRUE_W,
            'true_b': TRUE_B,
            'xs': list(x_np),
            'ys': list(y_np),
            'num_steps': n_steps}

    mess = dumps(mess)
    await ws.send(mess)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(n_steps):
            feed_dict = {x: x_np, y: y_np}
            _, summary, loss = sess.run([train_op, merged, l], feed_dict = feed_dict)
            train_writer.add_summary(summary, i)
            
            mess = {'type': 'STEP',
                    'num': i+1,
                    'w': float(w.eval()[0]),
                    'b': float(b.eval()[0]),
                    'loss': float(loss)}
            
            mess = dumps(mess)
            await ws.send(mess)
        



async def main(ws, path):
    
    global num_con
    mynum = num_con
    num_con += 1
    print(f'Connection #{num_con} started\n')
    train_data = make_data()
    
    try:
        await model(train_data['x'], train_data['y'], ws)
    except websockets.exceptions.ConnectionClosed:
        print(f'Connection #{num_con} ended abnormally\n')
        return
        
    print(f'Connection #{num_con} ended\n')



def run():
    
    global num_con
    num_con = 0
    start_server = websockets.serve(main, 'localhost', 9999)
    loop = asyncio.get_event_loop()
    loop.run_until_complete(start_server)
        
    m = 'Ready For Connections...'
    o = '-' * len(m)
    print(f'\n{o}\n{m}\n{o}\n')

    loop.run_forever()
    

if __name__ == '__main__':
    run()
