/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package deepNN;

import deepNN.activation.ActivationFunction;
import deepNN.activation.ReluFunction;
import deepNN.activation.SigmoidFunction;
import deepNN.activation.SoftmaxFunction;
import deepNN.loss.BinaryCrossEntropyLoss;
import deepNN.loss.LossFunction;
import deepNN.loss.MultiClassCrossEntropyLoss;
import utils.MLUtils;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Deep Neural Network with L hidden layers
 * 
 * @author Matias Leone
 */
public class DeepNeuralNetwork {

    public static final ActivationFunction RELU = new ReluFunction();
    public static final ActivationFunction SIGMOID = new SigmoidFunction();
    public static final ActivationFunction SOFTMAX = new SoftmaxFunction();
    public static final LossFunction BINARY_CROSS_ENTROPY = new BinaryCrossEntropyLoss();
    public static final LossFunction MULTI_CLASS_CROSS_ENTROPY = new MultiClassCrossEntropyLoss();

    private final int[] layerDims;
    private final long randSeed;
    private final int miniBatchSize;
    private final int iterations;
    private final float learningRate;
    private final float lambda;
    private final ActivationFunction hiddenActivationFunc;
    private final ActivationFunction outputActivationFunc;
    private final LossFunction lossFunction;
    private Map<String, Matrix2> parameters;

    /**
     * Creates a new neural network
     * @param randSeed random seed
     * @param layerDims array of dimensions for each layer (including input, hidden and output layers)
     * @param miniBatchSize size of mini-batches using for gradient descent
     * @param iterations number of epochs to run gradient descent on all mini-batches
     * @param learningRate learning rate (alpha) for gradient descent
     * @param lambda L2 regularization (lambda) for gradient descent
     * @param hiddenActivationFunc activation function for all hidden layers
     * @param outputActivationFunc activation function for the output layer
     * @param lossFunction loss function for the output layer
     */
    public DeepNeuralNetwork(long randSeed, int[] layerDims, int miniBatchSize, int iterations, float learningRate,
                             float lambda, ActivationFunction hiddenActivationFunc, ActivationFunction outputActivationFunc,
                             LossFunction lossFunction) {
        this.layerDims = layerDims;
        this.randSeed = randSeed;
        this.miniBatchSize = miniBatchSize;
        this.iterations = iterations;
        this.learningRate = learningRate;
        this.lambda = lambda;
        this.hiddenActivationFunc = hiddenActivationFunc;
        this.outputActivationFunc = outputActivationFunc;
        this.lossFunction = lossFunction;
    }

    /**
     * Tran the given samples
     * @param X features
     * @param Y labels
     * @param printCost true if you want to print the current cost in each iteration
     */
    public void train(Matrix2 X, Matrix2 Y, boolean printCost) {
        //Initialize parameters
        long currentSeed = randSeed;
        this.parameters = initializeParameters(this.layerDims, currentSeed);
        
        //Gradient descent loop
        List<CacheItem> caches = new ArrayList<>(this.layerDims.length - 1);
        Map<String, Matrix2> grads = new HashMap<>(this.layerDims.length - 1);
        List<MiniBatch> miniBatches = new ArrayList<>(X.cols() / this.miniBatchSize + 1);
        for (int i = 0; i < iterations; i++) {
            grads.clear();
            miniBatches.clear();
            
            //Create mini-batches
            currentSeed += 1;
            randomMiniBatches(X, Y, this.miniBatchSize, currentSeed, miniBatches);
            
            //Loop through all mini-batches
            float cost = Float.MAX_VALUE;
            for (MiniBatch miniBatch : miniBatches) {
                caches.clear();
                
                //Forward propagation
                Matrix2 AL = modelForward(miniBatch.X, this.parameters, caches, this.hiddenActivationFunc, this.outputActivationFunc);

                //Compute cost
                cost = computeCost(AL, miniBatch.Y, this.lambda, this.parameters, this.lossFunction);

                //Backward propagation
                grads = modelBackward(AL, miniBatch.Y, caches, grads, this.lambda, this.lossFunction, this.hiddenActivationFunc, this.outputActivationFunc);

                //Update parameters
                updateParameters(this.parameters, grads, this.learningRate);
            }
            
            //Print cost
            if(printCost) {
                if(i % 100 == 0) {
                    System.out.println("Cost after iteration " + i + ": " + cost);
                }
            } 
                      
        }
    }
    
    /**
     * Predict Y for the given X using the trained model
     */
    public Matrix2 predict(Matrix2 X) {
        List<CacheItem> caches = new ArrayList<>();
        
        Matrix2 AL = modelForward(X, parameters, caches, this.hiddenActivationFunc, this.outputActivationFunc);

        Matrix2 prediction;
        if(AL.rows() == 1) {
            //AL > 0.5
            prediction = AL.greater(0.5f);
        } else {
            //Convert each column of AL to a one-hot vec picking the max value
            Matrix2 max = AL.maxPerColumn().broadcastRow(AL.rows());
            prediction = Matrix2.eqEW(AL, max);
        }

        return prediction;
    }
    
    
    
    
    
    /**
     * Init W and b parameters for all layers
     */
    private Map<String, Matrix2> initializeParameters(int[] layerDims, long randSeed) {
        Map<String, Matrix2> parameters = new HashMap<>((layerDims.length - 1) * 2);
        for (int l = 1; l < layerDims.length; l++) {
            String layerIdx = String.valueOf(l);
            int rows = layerDims[l];
            int cols = layerDims[l - 1];
            parameters.put("W" + layerIdx, Matrix2.random(rows, cols, randSeed).mul(0.01f));
            parameters.put("b" + layerIdx, Matrix2.zeros(rows, 1));
        }
        return parameters;
    }
    
    /**
     * Forward propagation for all layers.
     * Compute AL and store intermediate values in caches
     */
    private Matrix2 modelForward(Matrix2 X, Map<String, Matrix2> parameters, List<CacheItem> caches,
                                 ActivationFunction hiddenActivation, ActivationFunction outputActivation) {
        Matrix2 A = X;
        int L = parameters.size() / 2;
        
        //Linear-Activation pass for all layers except the last one
        for (int l = 1; l < L; l++) {
            Matrix2 Aprev = A;
            String layerIdx = String.valueOf(l);
            Matrix2 W = parameters.get("W" + layerIdx);
            Matrix2 b = parameters.get("b" + layerIdx);
            A = linearActivationForward(Aprev, W, b, hiddenActivation, caches);
        }
        
        //Linear-Activation for last layer
        Matrix2 WL = parameters.get("W" + L);
        Matrix2 bL = parameters.get("b" + L);
        Matrix2 AL = linearActivationForward(A, WL, bL, outputActivation, caches);
        
        return AL;
    }
    
    /**
     * Activation and linear forward pass: A = g(Z)
     */
    private Matrix2 linearActivationForward(Matrix2 A_prev, Matrix2 W, Matrix2 b, ActivationFunction activation, List<CacheItem> caches) {
        Matrix2 Z = linearForward(A_prev, W, b);
        LinearCache linearCache = new LinearCache(A_prev, W, b);
        
        Matrix2 A = activation.forward(Z);
        ActivationCache activationCache = new ActivationCache(Z);
        
        caches.add(new CacheItem(linearCache, activationCache));
        return A;
    }
    
    /**
     * Linear forward pass: Z = W * A + b
     */
    private Matrix2 linearForward(Matrix2 A, Matrix2 W, Matrix2 b) {
        //Z = W * A + b;
        Matrix2 WxA = W.mul(A);
        Matrix2 Z = WxA.add(b.broadcastCol(WxA.cols()));
        return Z;
    }
    
    /**
     * Compute loss
     */
    private float computeCost(Matrix2 AL, Matrix2 Y, float lambda, Map<String, Matrix2> parameters, LossFunction lossFunction) {
        int m = Y.cols();
        int L = parameters.size() / 2;

        //Use loss function to compute cost
        float crossEntropyCost = lossFunction.computeCost(Y, AL);
        
        //L2 regularization cost: lambda/2m * (sum(W1^2) + sum(W2^2) + ... + (WL^2))
        float l2RegCost = 0;
        for (int l = 1; l < L; l++) {
            Matrix2 W = parameters.get("W" + l);
            l2RegCost += W.square().sum();
        }
        l2RegCost *= lambda / (2f * m);
        
        //Combined cost
        float cost = crossEntropyCost + l2RegCost;
        
        return cost;
    }
    
    /**
     * Backward propagation for all layers
     */
    private Map<String, Matrix2> modelBackward(Matrix2 AL, Matrix2 Y, List<CacheItem> caches, Map<String, Matrix2> grads,
                                               float lambda, LossFunction lossFunction, ActivationFunction hiddenActivation,
                                               ActivationFunction outputActivation) {
        int L = caches.size();
        CacheItem cache;
        String layerIdx;
        BackpropResult res;

        //Compute loss function gradient
        Matrix2 dAL = lossFunction.computeCostGradient(Y, AL);

        //Compute gradient for output layer
        cache = caches.get(L - 1);
        res = linearActivationBackward(dAL, cache, outputActivation, lambda);
        layerIdx = String.valueOf(L);
        grads.put("dA" + layerIdx, res.dA);
        grads.put("dW" + layerIdx, res.dW);
        grads.put("db" + layerIdx, res.db);
        
        //Compute gradients for all other layers
        for (int l = L - 2; l >= 0; l--) {
            layerIdx = String.valueOf(l + 1);
            cache = caches.get(l);
            Matrix2 dA_current = grads.get("dA" + (l + 2));
            res = linearActivationBackward(dA_current, cache, hiddenActivation, lambda);
            grads.put("dA" + layerIdx, res.dA);
            grads.put("dW" + layerIdx, res.dW);
            grads.put("db" + layerIdx, res.db);
        }
        
        return grads;
    }

    /**
     * Backward propagation for activation and linear
     */
    private BackpropResult linearActivationBackward(Matrix2 dA, CacheItem cache, ActivationFunction activation, float lambda) {
        Matrix2 dZ = activation.backward(dA, cache.activationCache.Z);
        return linearBackward(dZ, cache.linearCache, lambda); 
    }

    /**
     * Perform linear backward propagation
     */
    private BackpropResult linearBackward(Matrix2 dZ, LinearCache cache, float lambda) {
        int m = cache.Aprev.cols();
        
        //dW = 1/m * mul(dZ, Aprev.T) + lambda/m * W
        Matrix2 dW = dZ.mul(cache.Aprev.transpose()).mul(1f/m).add(cache.W.mul(lambda / m));
        
        //db = 1/m * sumCols(dZ)
        Matrix2 db = dZ.sumColumns().mul(1f/m);
        
        //dAprev = mul(W.T, dZ)
        Matrix2 dAprev = cache.W.transpose().mul(dZ);
        
        return new BackpropResult(dAprev, dW, db);
    }

    /**
     * Update parameters using gradient
     */
    private void updateParameters(Map<String, Matrix2> parameters, Map<String, Matrix2> grads, float learningRate) {
        int L = parameters.size() / 2;
        for (int l = 1; l <= L; l++) {
            String layerIdx = String.valueOf(l);
            Matrix2 W = parameters.get("W" + layerIdx);
            Matrix2 b = parameters.get("b" + layerIdx);
            Matrix2 dW = grads.get("dW" + layerIdx);
            Matrix2 db = grads.get("db" + layerIdx);
            
            //W = W - learningRate * dW
            W = W.sub(dW.mul(learningRate));
            b = b.sub(db.mul(learningRate));
            
            parameters.put("W" + layerIdx, W);
            parameters.put("b" + layerIdx, b);
        }
    }

    /**
     * Split input into random mini-batches. The miniBatches is populated.
     */
    public void randomMiniBatches(Matrix2 X, Matrix2 Y, int miniBatchSize, long randSeed, List<MiniBatch> miniBatches) {
        //Shuffle sample indices
        int m = Y.cols();
        int[] indices = MLUtils.shuffleArray(m, randSeed);
        
        //Assembly complete mini-batches
        int completeBatches = m / miniBatchSize;
        int[] batchIndices = new int[miniBatchSize];
        for (int i = 0; i < completeBatches; i++) {
            System.arraycopy(indices, i * miniBatchSize, batchIndices, 0, batchIndices.length);
            Matrix2 batchX = Matrix2.getColumns(X, batchIndices);
            Matrix2 batchY = Matrix2.getColumns(Y, batchIndices);
            miniBatches.add(new MiniBatch(batchX, batchY));
        }
        
        //Assembly last incomplete batch
        int pendingSize = m % miniBatchSize;
        if(pendingSize != 0) {
            batchIndices = new int[pendingSize];
            System.arraycopy(indices, completeBatches * miniBatchSize, batchIndices, 0, batchIndices.length);
            Matrix2 batchX = Matrix2.getColumns(X, batchIndices);
            Matrix2 batchY = Matrix2.getColumns(Y, batchIndices);
            miniBatches.add(new MiniBatch(batchX, batchY));
        }
    }
    
    private static class CacheItem {
        public final LinearCache linearCache;
        public final ActivationCache activationCache;
        public CacheItem(LinearCache linearCache, ActivationCache activationCache) {
            this.linearCache = linearCache;
            this.activationCache = activationCache;
        }
    }
    
    private static class LinearCache {
        public final Matrix2 Aprev;
        public final Matrix2 W;
        public final Matrix2 b;
        public LinearCache(Matrix2 Aprev, Matrix2 W, Matrix2 b) {
            this.Aprev = Aprev;
            this.W = W;
            this.b = b;
        }
    }
    
    private static class ActivationCache {
        public final Matrix2 Z;
        public ActivationCache(Matrix2 Z) {
            this.Z = Z;
        }
    }
    
    private static class BackpropResult {
        public final Matrix2 dA;
        public final Matrix2 dW;
        public final Matrix2 db;
        public BackpropResult(Matrix2 dA, Matrix2 dW, Matrix2 db) {
            this.dA = dA;
            this.dW = dW;
            this.db = db;
        }  
    }
    
    private static class MiniBatch {
        public final Matrix2 X;
        public final Matrix2 Y;
        public MiniBatch(Matrix2 X, Matrix2 Y) {
            this.X = X;
            this.Y = Y;
        }
        
    }


    
    
    
}
