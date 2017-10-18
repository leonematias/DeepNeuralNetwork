package deepNN.loss;

import deepNN.Matrix2;

/**
 * Loss function interface
 *
 * Created by matias.leone on 10/18/17.
 */
public interface LossFunction {

    float computeCost(Matrix2 Y, Matrix2 AL);

    Matrix2 computeCostGradient(Matrix2 Y, Matrix2 AL);

}
