package utils;

import java.util.List;
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
        Random rand = new Random(randSeed);
        for (SampleItem item : items) {
            if(rand.nextFloat() < splitPercentage) {
                out1.add(item);
            } else {
                out2.add(item);
            }
        }
    }


}
