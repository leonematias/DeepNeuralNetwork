package deepNN.loss;

import deepNN.Matrix2;

/**
 * Created by matias.leone on 10/18/17.
 */
public class MultiClassCrossEntropyLoss implements LossFunction {

    @Override
    public float computeCost(Matrix2 Y, Matrix2 AL) {
        int m = Y.cols();

        //Cost = -1/m * sum(Y * log(AL))
        float cost = -1f/m * Y.mulEW(AL.clampToZero().log()).sum();

        return cost;
    }

    @Override
    public Matrix2 computeCostGradient(Matrix2 Y, Matrix2 AL) {
        //Grad = AL - Y
        return AL.sub(Y);
    }
}
