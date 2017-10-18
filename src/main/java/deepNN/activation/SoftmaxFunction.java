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
        //A = exp(Z - Max) / (sum(exp(Z - Max)))
        Matrix2 max = Z.maxRows().broadcastRow(Z.rows());
        Matrix2 expZ = Z.sub(max).exp();
        return expZ.div(expZ.sum());
    }

    @Override
    public Matrix2 backward(Matrix2 dA, Matrix2 Z) {
        //S = softmax(Z)
        Matrix2 S = forward(Z);

        //dZ = dA * S * (1 - S)
        Matrix2 dZ = dA.mulEW(S).mulEW(S.oneMinus());

        return dZ;
    }
}
