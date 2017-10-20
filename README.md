# DeepNeuralNetwork
Deep neural network implemented in Java from scratch.
Plan Java code without using any library or framework.
Idea for those who are beginning with machine learning.
Based on guidelines by Andrew NG in his Deep Learning specialization at Cousera.

Features:
* Muli-layer network, the number of layers is configurable.
* Binary and Multi-class classification.
* RELU as activation function for hidden layers and Sigmoid/Softmax for output layer.
* Vectorized implementation.
* L2 regularization.
* Gradient descent with mini-batches.

Examples:
* ExampleMnistBinaryClassifier: distinguish between 0s and 1s digit images from Mnist.
* ExampleMnistMultiClassClassifier: classify Mnist digit images from 0 to 9.

This implementation uses Matrix2 class to perform linear algebra operations. That class was design for academic purposes and it's not optimized in any way. 

