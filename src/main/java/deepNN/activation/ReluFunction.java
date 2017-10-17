package deepNN.activation;

import deepNN.Matrix2;

/**
 * RELU activation function
 *
 * Created by matias.leone on 10/17/17.
 */
public class ReluFunction implements ActivationFunction {

    @Override
    public Matrix2 forward(Matrix2 Z) {
        return Z.relu();
    }

    @Override
    public Matrix2 backward(Matrix2 dA, Matrix2 Z) {
        //Create mask where all values <=0 are 0
        Matrix2 mask = Z.greater(0f);

        //dz = 0 if z <= 0 else keep value of da
        return dA.mulEW(mask);
    }
}
