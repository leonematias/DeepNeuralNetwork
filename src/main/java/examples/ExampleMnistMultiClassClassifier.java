/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package examples;

import deepNN.DeepNeuralNetwork;
import deepNN.Matrix2;
import utils.MLUtils;
import utils.MnistLoader;
import utils.PredictionStats;
import utils.SampleItem;

import java.util.ArrayList;
import java.util.List;

/**
 * Use Mnist data to train multi-class classifier to predict images digits from 0 to 9
 * 
 * @author matias.leone
 */
public class ExampleMnistMultiClassClassifier {
    
    public static void main(String[] args) {
        new ExampleMnistMultiClassClassifier().run();
    }
    
    private void run() {
        long randSeed = 12345;
        
        //Load data
        int labelsCount = 10;
        List<SampleItem> allImageData = MnistLoader.loadMnistData();
        
        //Split train and test set
        List<SampleItem> trainSet = new ArrayList<>();
        List<SampleItem> testSet = new ArrayList<>();
        MLUtils.splitDataSet(allImageData, 0.7f, randSeed, trainSet, testSet);
        
        //Convert to X,Y matrices. Use one-hot vector for Y labels
        Matrix2 trainX = SampleItem.toX(trainSet);
        Matrix2 trainY = SampleItem.toYoneHot(trainSet, labelsCount);
        Matrix2 testX = SampleItem.toX(testSet);
        Matrix2 testY = SampleItem.toYoneHot(testSet, labelsCount);
        
        
        //Train multi-class classifier with layers [400, 25, 10, 10]
        DeepNeuralNetwork classifier = new DeepNeuralNetwork(
                randSeed,
                new int[]{MnistLoader.MNIST_IMG_WIDTH * MnistLoader.MNIST_IMG_HEIGHT, 10, 10, labelsCount}, //network layers
                64, //mini-batch size
                1000, //epochs
                0.075f, //learning rate
                0.7f, //L2 lambda regularization,
                DeepNeuralNetwork.RELU, //Hidden layers activation function
                DeepNeuralNetwork.SOFTMAX, //Output layer activation function
                DeepNeuralNetwork.MULTI_CLASS_CROSS_ENTROPY //Loss function
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
