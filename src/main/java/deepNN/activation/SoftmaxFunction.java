package deepNN.activation;

import deepNN.Matrix2;

/**
 * Softmax activation function
 *
 * Created by matias.leone on 10/17/17.
 */
public class SoftmaxFunction implements ActivationFunction {

    @Override
    public Matrix2 forward(Matrix2 Z) {
        //A = exp(Z) / (1 + sum(exp(Z)))
        Matrix2 expZ = Z.exp();
        return expZ.div(1f + expZ.sum());
    }

    @Override
    public Matrix2 backward(Matrix2 dA, Matrix2 Z) {
        return null;
    }
}
