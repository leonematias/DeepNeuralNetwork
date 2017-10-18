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
import java.util.Map;

/**
 * Use Mnist data to train binary classifier to predict images of 0 and 1 digits
 * 
 * @author matias.leone
 */
public class ExampleMnistBinaryClassifier {
    
    public static void main(String[] args) {
        new ExampleMnistBinaryClassifier().run();
    }
    
    private void run() {
        long randSeed = 12345;
        
        //Load data
        List<SampleItem> allImageData = MnistLoader.loadMnistData();
        
        //Pick 0 and 1 images
        Map<Integer, List<SampleItem>> imageMap = SampleItem.toMap(allImageData);
        List<SampleItem> digitImages = new ArrayList<>();
        digitImages.addAll(imageMap.get(0));
        digitImages.addAll(imageMap.get(1));
        
        
        //Split train and test set
        List<SampleItem> trainSet = new ArrayList<>();
        List<SampleItem> testSet = new ArrayList<>();
        MLUtils.splitDataSet(digitImages, 0.7f, randSeed, trainSet, testSet);
        
        //Convert to X,Y matrices
        Matrix2 trainX = SampleItem.toX(trainSet);
        Matrix2 trainY = SampleItem.toY(trainSet);
        Matrix2 testX = SampleItem.toX(testSet);
        Matrix2 testY = SampleItem.toY(testSet);
        
        
        //Train binary classifier with layers [400, 25, 10, 1]
        DeepNeuralNetwork classifier = new DeepNeuralNetwork(
                randSeed,
                new int[]{MnistLoader.MNIST_IMG_WIDTH * MnistLoader.MNIST_IMG_HEIGHT, 25, 10, 1}, //network layers
                128, //mini-batch size
                1000, //epochs
                0.075f, //learning rate
                0.7f, //L2 lambda regularization,
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
