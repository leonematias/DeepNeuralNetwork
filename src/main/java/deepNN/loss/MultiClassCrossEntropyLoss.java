package deepNN.loss;

import deepNN.Matrix2;

/**
 * Created by matias.leone on 10/18/17.
 */
public class MultiClassCrossEntropyLoss implements LossFunction {

    @Override
    public float computeCost(Matrix2 Y, Matrix2 AL) {
        int m = Y.cols();
        int K = Y.rows();

        //Cost = -1/m * sum(-Y_k * log(AL_k))
        float cost = 0;
        for (int k = 0; k < K; k++) {
            Matrix2 Yk = Matrix2.getRow(Y, k);
            Matrix2 ALk = Matrix2.getRow(AL, k);
            cost += Yk.mulEW(ALk.log()).sum();
        }
        cost *= -1f/m;

        return cost;
    }

    @Override
    public Matrix2 computeCostGradient(Matrix2 Y, Matrix2 AL) {
        //Grad = Y - AL
        return Y.sub(AL);
    }
}
