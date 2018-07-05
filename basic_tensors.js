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
