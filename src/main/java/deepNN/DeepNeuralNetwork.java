/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package deepNN;

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
    
    private final int[] layerDims;
    private final long randSeed;
    private final int miniBatchSize;
    private final int iterations;
    private final float learningRate;
    private final float lambda;
    private Map<String, Matrix2> parameters;
    
    public DeepNeuralNetwork(long randSeed, int[] layerDims, int miniBatchSize, int iterations, float learningRate, float lambda) {
        this.layerDims = layerDims;
        this.randSeed = randSeed;
        this.miniBatchSize = miniBatchSize;
        this.iterations = iterations;
        this.learningRate = learningRate;
        this.lambda = lambda;
    }

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
                Matrix2 AL = modelForward(miniBatch.X, this.parameters, caches);

                //Compute cost
                cost = computeCost(AL, miniBatch.Y, this.lambda, this.parameters);

                //Backward propagation
                grads = modelBackward(AL, miniBatch.Y, caches, grads, this.lambda);

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
        
        Matrix2 AL = modelForward(X, parameters, caches);
        
        //AL > 0.5
        return AL.greater(0.5f);
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
    private Matrix2 modelForward(Matrix2 X, Map<String, Matrix2> parameters, List<CacheItem> caches) {
        Matrix2 A = X;
        int L = parameters.size() / 2;
        
        //Linear-Relu pass for all layers except the last one
        for (int l = 1; l < L; l++) {
            Matrix2 Aprev = A;
            String layerIdx = String.valueOf(l);
            Matrix2 W = parameters.get("W" + layerIdx);
            Matrix2 b = parameters.get("b" + layerIdx);
            A = linearActivationForward(Aprev, W, b, Matrix2.ReluOp.INSTANCE, caches);
        }
        
        //Linear-Sigmoid for last layer
        Matrix2 WL = parameters.get("W" + L);
        Matrix2 bL = parameters.get("b" + L);
        Matrix2 AL = linearActivationForward(A, WL, bL, Matrix2.SigmoidOp.INSTANCE, caches);
        
        return AL;
    }
    
    /**
     * Activation and linear forward pass: A = g(Z)
     */
    private Matrix2 linearActivationForward(Matrix2 A_prev, Matrix2 W, Matrix2 b, Matrix2.ElementWiseOp activation, List<CacheItem> caches) {
        Matrix2 Z = linearForward(A_prev, W, b);
        LinearCache linearCache = new LinearCache(A_prev, W, b);
        
        Matrix2 A = Z.apply(activation);
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
    private float computeCost(Matrix2 AL, Matrix2 Y, float lambda, Map<String, Matrix2> parameters) {
        int m = Y.cols();
        int L = parameters.size() / 2;
        
        //Cross-entropy cost = -1/m * sum(Y * log(AL) + (1-Y) * log(1-AL))
        float crossEntropyCost = Matrix2.add(
                Matrix2.mulEW(Y, AL.log()), 
                Matrix2.mulEW(Y.oneMinus(), AL.oneMinus().log())
        ).sumColumns().mul(-1f/m).get(0,0);
        
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
    private Map<String, Matrix2> modelBackward(Matrix2 AL, Matrix2 Y, List<CacheItem> caches, Map<String, Matrix2> grads, float lambda) {
        int m = Y.cols();
        int L = caches.size();
        CacheItem cache;
        String layerIdx;
        BackpropResult res;

        //Cross-entropy loss derivative: dAL = - ((Y / AL) - ((1-Y) / (1-AL)))
        Matrix2 dAL = Matrix2.divEW(Y, AL).sub(Matrix2.divEW(Y.oneMinus(), AL.oneMinus())).mul(-1f);

        //Compute sigmoid gradient for output layer
        cache = caches.get(L - 1);
        res = linearActivationBackward(dAL, cache, SigmoidBackward.INSTANCE, lambda);
        layerIdx = String.valueOf(L);
        grads.put("dA" + layerIdx, res.dA);
        grads.put("dW" + layerIdx, res.dW);
        grads.put("db" + layerIdx, res.db);
        
        //Compute relu gradients for all other layers
        for (int l = L - 2; l >= 0; l--) {
            layerIdx = String.valueOf(l + 1);
            cache = caches.get(l);
            Matrix2 dA_current = grads.get("dA" + (l + 2));
            res = linearActivationBackward(dA_current, cache, ReluBackward.INSTANCE, lambda);
            grads.put("dA" + layerIdx, res.dA);
            grads.put("dW" + layerIdx, res.dW);
            grads.put("db" + layerIdx, res.db);
        }
        
        return grads;
    }
    
    private BackpropResult linearActivationBackward(Matrix2 dA, CacheItem cache, BackwardOp activation, float lambda) {
        Matrix2 dZ = activation.apply(dA, cache.activationCache);    
        return linearBackward(dZ, cache.linearCache, lambda); 
    }
    
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
    
    interface BackwardOp {
        Matrix2 apply(Matrix2 dA, ActivationCache cache);
    }
    
    private static class ReluBackward implements BackwardOp {
        public static final BackwardOp INSTANCE = new ReluBackward();
        @Override
        public Matrix2 apply(Matrix2 dA, ActivationCache cache) {            
            //Create mask where all values <=0 are 0
            Matrix2 mask = cache.Z.greater(0f);

            //dz = 0 if z <= 0 else keep value of da
            return dA.mulEW(mask);
        } 
    }
    
    private static class SigmoidBackward implements BackwardOp {
        public static final BackwardOp INSTANCE = new SigmoidBackward();
        @Override
        public Matrix2 apply(Matrix2 dA, ActivationCache cache) {
            //S = 1 / (1 + e^(-Z))
            Matrix2 S = cache.Z.sigmoid();

            //dZ = dA * s * (1-s)
            Matrix2 dZ = dA.mulEW(S).mulEW(S.oneMinus());

            return dZ;
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
