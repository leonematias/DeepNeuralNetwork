package utils;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/**
 * Common machine learning utils
 */
public abstract class MLUtils {

    private MLUtils(){}

    public static int[] shuffleArray(int n, long randSeed) {
        int[] a = new int[n];
        for (int i = 0; i < n; i++) {
            a[i] = i;
        }
        Random rand = new Random(randSeed);
        for (int i = 0; i < n; i++) {
            int j = i + rand.nextInt(n - i);
            int tmp = a[i];
            a[i] = a[j];
            a[j] = tmp;
        }
        return a;
    }

    public static void splitDataSet(List<SampleItem> items, float splitPercentage, long randSeed, List<SampleItem> out1, List<SampleItem> out2) {
        splitDataSet(SampleItem.toMap(items), splitPercentage, randSeed, out1, out2);
    }

    public static void splitDataSet(Map<Integer, List<SampleItem>> itemsPerLabel, float splitPercentage, long randSeed, List<SampleItem> out1, List<SampleItem> out2) {
        Random rand = new Random(randSeed);
        for (int label : itemsPerLabel.keySet()) {
            for (SampleItem item : itemsPerLabel.get(label)) {
                if(rand.nextFloat() < splitPercentage) {
                    out1.add(item);
                } else {
                    out2.add(item);
                }
            }
        }
    }

    public static int minSamplesCount(Map<Integer, List<SampleItem>> itemsPerLabel) {
        int min = Integer.MAX_VALUE;
        for (List<SampleItem> items : itemsPerLabel.values()) {
            min = Math.min(min, items.size());
        }
        return min;
    }

    public static float[] oneHotVec(int label, int totalLabels) {
        float[] vec = new float[totalLabels];
        vec[label] = 1;
        return vec;
    }

    public static Map<Integer, Float> computeSamplesDiversity(List<SampleItem> items) {
        int m = items.size();
        Map<Integer, Float> labelMap = new HashMap<>();
        for (SampleItem item : items) {
            Float currentObj = labelMap.get(item.getLabel());
            float total = 0;
            if(currentObj != null) {
                total = currentObj;
            }
            labelMap.put(item.getLabel(), total + 1);
        }
        for (Map.Entry<Integer, Float> e : labelMap.entrySet()) {
            e.setValue(e.getValue() / m * 100f);
        }
        return labelMap;
    }

    public static void printSamplesDiversity(List<SampleItem> items) {
        Map<Integer, Float> diversity = computeSamplesDiversity(items);
        for (Map.Entry<Integer, Float> e : diversity.entrySet()) {
            System.out.println(" - " + e.getKey() + ": " + e.getValue() + "%");
        }
    }


}
