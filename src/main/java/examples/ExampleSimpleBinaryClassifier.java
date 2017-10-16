/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package examples;

import deepNN.DeepNeuralNetwork;
import deepNN.Matrix2;
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

        //Small network test case
        List<SampleItem> trainSet = new ArrayList<>();
        Random rand = new Random(randSeed);
        for (int i = 0; i < 1000; i++) {
            float px = -1 + rand.nextFloat() * 2;
            float py = -1 + rand.nextFloat() * 2;
            int label = px < 0 ? 0 : 1;
            trainSet.add(new SampleItem(new float[]{px, py}, label));
        }
        List<SampleItem> testSet = new ArrayList<>(Arrays.asList(
                new SampleItem(new float[]{-0.1f, -1}, 0),
                new SampleItem(new float[]{-0.7f, 1}, 0),
                
                new SampleItem(new float[]{0.2f, -1}, 1),
                new SampleItem(new float[]{0.9f, 0.5f}, 1)
        ));

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
                1000, //epochs
                0.075f, //learning rate
                0.7f //L2 lambda regularization
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
