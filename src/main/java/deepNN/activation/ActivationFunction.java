package deepNN.activation;

import deepNN.Matrix2;

/**
 * Interface for an activation function
 *
 * Created by matias.leone on 10/17/17.
 */
public interface ActivationFunction {

    Matrix2 forward(Matrix2 Z);

    Matrix2 backward(Matrix2 dA, Matrix2 Z);
}
