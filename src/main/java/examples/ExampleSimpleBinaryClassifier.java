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
 * Simple binary classifier example
 * 
 * @author matias.leone
 */
public class ExampleSimpleBinaryClassifier {
    
    public static void main(String[] args) {
        new ExampleSimpleBinaryClassifier().run();
    }
    
    private void run() {
        long randSeed = 12345;
        int samplesCount = 10000;

        /**
         * Create N random points between (-1, 1) distributed in 2 groups:
         * 0: x < 0
         * 1: x > 0
         */
        List<SampleItem> trainSet = new ArrayList<>();
        Random rand = new Random(randSeed);
        for (int i = 0; i < samplesCount; i++) {
            float px = -1 + rand.nextFloat() * 2;
            float py = -1 + rand.nextFloat() * 2;
            int label = px < 0 ? 0 : 1;
            trainSet.add(new SampleItem(new float[]{px, py}, label));
        }

        //Test set: a few samples of each group
        List<SampleItem> testSet = new ArrayList<>(Arrays.asList(
                //0: x < 0
                new SampleItem(new float[]{-0.1f, -1}, 0),
                new SampleItem(new float[]{-0.7f, 1}, 0),

                //1: x > 0
                new SampleItem(new float[]{0.2f, -1}, 1),
                new SampleItem(new float[]{0.9f, 0.5f}, 1)
        ));

        //Print distribution of samples in both sets
        System.out.println("Train set diversity:");
        MLUtils.printSamplesDiversity(trainSet);
        System.out.println("Test set diversity:");
        MLUtils.printSamplesDiversity(testSet);

        //Convert to X,Y matrices
        Matrix2 trainX = SampleItem.toX(trainSet);
        Matrix2 trainY = SampleItem.toY(trainSet);
        Matrix2 testX = SampleItem.toX(testSet);
        Matrix2 testY = SampleItem.toY(testSet);

        //Train binary classifier
        DeepNeuralNetwork classifier = new DeepNeuralNetwork(
                randSeed,
                new int[]{2, 1}, //network layers
                128, //mini-batch size
                2000, //epochs
                0.075f, //learning rate
                0.7f, //L2 lambda regularization
                DeepNeuralNetwork.RELU, //Hidden layers activation function
                DeepNeuralNetwork.SIGMOID, //Output layer activation function
                DeepNeuralNetwork.BINARY_CROSS_ENTROPY //Loss function
        );
        classifier.train(trainX, trainY, true);

        //Predict train and test set
        Matrix2 trainYpred = classifier.predict(trainX);
        Matrix2 testYpred = classifier.predict(testX);
        PredictionStats trainStats = new PredictionStats(trainY, trainYpred);
        PredictionStats testStats = new PredictionStats(testY, testYpred);
        System.out.println("Train set performance: " + trainStats.toString());
        System.out.println("Test set performance: " + testStats.toString());
        
    }
    
}
