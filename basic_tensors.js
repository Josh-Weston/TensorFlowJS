// Tensors = Scalars, Vectors, Matrices

//Scalar = rank 0. Single value

//rank 1 = a vector (either column or row)
//featurizaton = is a representation of a real-world entity as a vector (or more generally as a tensor)
//Example = (height weight color) to represent a cat.
//It is often not necessary to invent a new featurization method. Similar to how we do not need to create
//new data structures, many popular items (pictures, molecules, etc.) already have well studies featurization
//methods.

//rank 2 = a matrix (rows X columns)
//matrix multiplication = mn * nk = mk
//rows of one matrix are multiplied by columns of another matrix

//scalar, we don't need to know the index (rank-0)
//vector, we need the index (rank-1)
//matrix, we need the row and column (rank-2)
//cube, we need the row, column, and depth (rank-3

// *** RANK = NUMBER OF INDICES REQUIRED TO FIND AN INDIVIDUAL VALUE!! ***

let c = tf.ones([2, 2]);
let d = tf.ones([2, 2]);

// Element wise
c.add(d).print();
c.mul(d).print();

//Identity matrix
tf.eye(5).print();

let range = tf.range(1, 5, 1)
range.print();

//tf.diag(range).print(); Not a function in tensorflow js

let ones = tf.ones([2, 3]);
ones.print();
tf.transpose(ones).print();

//Matrix multiplication
tf.matMul(ones, tf.transpose(ones)).print();

// Casting
tf.ones([3, 3]).toFloat().print();
tf.ones([3, 3]).toInt().print();
tf.ones([3, 3]).toBool().print();

tf.reshape(tf.ones([8]), [2, 4]).print();

console.log(tf.ones([3, 3]).shape);
console.log(tf.ones([3, 3]).rank);
console.log(tf.expandDims(tf.ones([3, 3]), 1).rank);

/* Broadcasting */
a = tf.ones([2, 2], dtype="float32");
b = tf.range(0, 2, 1, dtype="float32");

a.print();
b.print();

c.add(b).print();

// Tensorflow is declarative (like SQL), not imperative (like normal programming)
// We can create new Tensors, but cannot manipulate the values in existing tensors
// Machine learning generally depends on stateful computations (e.g. changing variables)

// *** Learning algorithms are essentially rules for updating stored
// tensors to explain provided data. If it is not possible to update
// these stored tensors, it would be hard to learn. ***

let v = tf.variable(tf.ones([2, 2]));
v.print();
v.assign(tf.fill([2, 2], 5)); // Notice how we can change the values in the tensor through assign
v.print();

