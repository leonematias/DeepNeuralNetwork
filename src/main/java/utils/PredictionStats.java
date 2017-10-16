package utils;

import deepNN.Matrix2;

/**
 * Stats to evaluate prediction
 */
public class PredictionStats {
    private int truePositives;
    private int falsePositives;
    private int trueNegatives;
    private int falseNegatives;
    private float precision;
    private float recall;
    private float accuracy;
    private float f1;

    public PredictionStats(Matrix2 Y, Matrix2 Yhat) {
        int m = Y.cols();
        truePositives = 0;
        falsePositives = 0;
        trueNegatives = 0;
        falseNegatives = 0;
        for (int col = 0; col < m; col++) {
            float y = Y.get(0, col);
            float yhat = Yhat.get(0, col);
            if(y == 1 && yhat == 1) {
                truePositives++;
            } else if(y == 0 && yhat == 1) {
                falsePositives++;
            } else if(y == 1 && yhat == 0) {
                falseNegatives++;
            } else {
                trueNegatives++;
            }
        }
        precision = (float)truePositives / (truePositives + falsePositives);
        recall = (float)truePositives / (truePositives + falseNegatives);
        accuracy = (float)(truePositives + trueNegatives) / (truePositives + trueNegatives + falsePositives + falseNegatives);
        f1 = 2 * (precision * recall) / (precision + recall);
    }

    public int getTruePositives() {
        return truePositives;
    }

    public int getFalsePositives() {
        return falsePositives;
    }

    public int getTrueNegatives() {
        return trueNegatives;
    }

    public int getFalseNegatives() {
        return falseNegatives;
    }

    public float getPrecision() {
        return precision;
    }

    public float getRecall() {
        return recall;
    }

    public float getAccuracy() {
        return accuracy;
    }

    public float getF1() {
        return f1;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(50);
        sb.append("Accuracy: ").append(accuracy * 100f);
        sb.append(", Precision: ").append(precision * 100f);
        sb.append(", Recall: ").append(recall * 100f);
        sb.append(", F1: ").append(f1 * 100f);
        return sb.toString();
    }
}
