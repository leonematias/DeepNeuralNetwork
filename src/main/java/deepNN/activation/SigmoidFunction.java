package deepNN.activation;

import deepNN.Matrix2;

/**
 * Sigmoid activation function
 *
 * Created by matias.leone on 10/17/17.
 */
public class SigmoidFunction implements ActivationFunction {

    @Override
    public Matrix2 forward(Matrix2 Z) {
        return Z.sigmoid();
    }

    @Override
    public Matrix2 backward(Matrix2 dA, Matrix2 Z) {
        //S = 1 / (1 + e^(-Z))
        Matrix2 S = Z.sigmoid();

        //dZ = dA * s * (1-s)
        Matrix2 dZ = dA.mulEW(S).mulEW(S.oneMinus());

        return dZ;
    }
}
