package deepNN.loss;

import deepNN.Matrix2;

/**
 * Binary cross entropy loss function
 *
 * Created by matias.leone on 10/18/17.
 */
public class BinaryCrossEntropyLoss implements LossFunction {

    @Override
    public float computeCost(Matrix2 Y, Matrix2 AL) {
        int m = Y.cols();

        //Cross-entropy cost = -1/m * sum(Y * log(AL) + (1-Y) * log(1-AL))
        float cost = Matrix2.add(
                Matrix2.mulEW(Y, AL.clampToZero().log()),
                Matrix2.mulEW(Y.oneMinus(), AL.oneMinus().clampToZero().log())
        ).sumColumns().mul(-1f/m).get(0,0);

        return cost;
    }

    @Override
    public Matrix2 computeCostGradient(Matrix2 Y, Matrix2 AL) {
        //Cross-entropy loss derivative: dAL = - ((Y / AL) - ((1-Y) / (1-AL)))
        Matrix2 dAL = Matrix2.divEW(Y, AL).sub(Matrix2.divEW(Y.oneMinus(), AL.oneMinus())).mul(-1f);

        return dAL;
    }
}
