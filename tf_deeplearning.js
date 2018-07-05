// Prepopulating tensors.

let test = tf.zeros([2, 2]).print();

test = tf.ones([2, 2]).print();

test = tf.fill([2, 2], 10).print();

test = tf.randomNormal([4, 4]).print();

test = tf.randomUniform([4, 4], minval=50, maxval=100).print(); //Can pass argument names in ES6

test = tf.truncatedNormal([4, 4]).print(); //Drops anything more than 2 standard deviations from the mean to avoid instability


// Note: we randomize because update equations usually satisfy the property that weights initialized
// at the same value will continue to evolve tomorrow. If all weights are the same, the model can't
// learn much!