/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package examples;

import deepNN.DeepNeuralNetwork;
import deepNN.Matrix2;
import utils.MLUtils;
import utils.PredictionStats;
import utils.SampleItem;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;

/**
 * Simple multi-class classifier example
 * 
 * @author matias.leone
 */
public class ExampleSimpleMultiClassClassifier {
    
    public static void main(String[] args) {
        new ExampleSimpleMultiClassClassifier().run();
    }
    
    private void run() {
        long randSeed = 12345;
        int labelsCount = 4;

        //Small network test case
        List<SampleItem> trainSet = new ArrayList<>();
        Random rand = new Random(randSeed);
        for (int i = 0; i < 10000; i++) {
            float px = -1 + rand.nextFloat() * 2;
            float py = -1 + rand.nextFloat() * 2;
            int label = (px < 0 ? 0 : 1) + (py < 0 ? 0 : 2);
            trainSet.add(new SampleItem(new float[]{px, py}, label));
        }
        List<SampleItem> testSet = new ArrayList<>(Arrays.asList(
                new SampleItem(new float[]{-0.1f, -1}, 0),
                new SampleItem(new float[]{-0.3f, -0.4f}, 0),

                new SampleItem(new float[]{0.1f, -0.7f}, 1),
                new SampleItem(new float[]{0.3f, -0.9f}, 1),

                new SampleItem(new float[]{-0.1f, 1}, 2),
                new SampleItem(new float[]{-0.3f, 0.4f}, 2),

                new SampleItem(new float[]{0.1f, 0.7f}, 3),
                new SampleItem(new float[]{0.3f, 0.9f}, 3)
        ));

        System.out.println("Train set diversity:");
        MLUtils.printSamplesDiversity(trainSet);
        System.out.println("Test set diversity:");
        MLUtils.printSamplesDiversity(testSet);

        //Convert to X,Y matrices
        Matrix2 trainX = SampleItem.toX(trainSet);
        Matrix2 trainY = SampleItem.toYoneHot(trainSet, labelsCount);
        Matrix2 testX = SampleItem.toX(testSet);
        Matrix2 testY = SampleItem.toYoneHot(testSet, labelsCount);

        //Train binary classifier
        DeepNeuralNetwork classifier = new DeepNeuralNetwork(
                randSeed,
                new int[]{2, 10, 10, labelsCount}, //network layers
                128, //mini-batch size
                4000, //epochs
                0.075f, //learning rate
                0, //L2 lambda regularization
                DeepNeuralNetwork.RELU, //Hidden layers activation function
                DeepNeuralNetwork.SOFTMAX, //Output layer activation function
                DeepNeuralNetwork.MULTI_CLASS_CROSS_ENTROPY //Loss function
        );
        classifier.train(trainX, trainY, true);

        //Predict train and test set
        Matrix2 trainYpred = classifier.predict(trainX);
        Matrix2 testYpred = classifier.predict(testX);
        //PredictionStats trainStats = new PredictionStats(trainY, trainYpred);
        //PredictionStats testStats = new PredictionStats(testY, testYpred);
        System.out.println("Train set performance: " + PredictionStats.computeAccuracy(trainY, trainYpred) * 100f);
        System.out.println("Test set performance: " + PredictionStats.computeAccuracy(testY, testYpred) * 100f);
        
    }
    
}
