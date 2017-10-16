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
 
    private final static int MNIST_IMG_WIDTH = 20;
    private final static int MNIST_IMG_HEIGHT = 20;
    
    public static void main(String[] args) {
        new ExampleMnistBinaryClassifier().run();
    }
    
    private void run() {
        long randSeed = 12345;
        
        //Load data
        String xPath = "data/mnist_input_images.csv";
        String yPath = "data/mnist_input_classification.csv";
        List<SampleItem> allImageData = MnistLoader.loadMnistData(xPath, yPath);
        
        //Pick 0 and 1 images (same amount from both)
        Map<Integer, List<SampleItem>> imageMap = SampleItem.toMap(allImageData);
        List<SampleItem> zeroImages = imageMap.get(0);
        List<SampleItem> oneImages = imageMap.get(1);
        int minSampleSize = Math.min(zeroImages.size(), oneImages.size());
        List<SampleItem> digitImages = new ArrayList<>(minSampleSize * 2);
        for (int i = 0; i < minSampleSize; i++) {
            digitImages.add(zeroImages.get(i));
            digitImages.add(oneImages.get(i));
        }
        
        
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
                new int[]{MNIST_IMG_WIDTH * MNIST_IMG_HEIGHT, 25, 10, 1}, //network layers
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
