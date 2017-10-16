package utils;

import deepNN.Matrix2;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Represents one item to be used for training or testing.
 * It contains features and label
 */
public class SampleItem {
    private final float[] features;
    private final int label;

    public SampleItem(float[] features, int label) {
        this.features = features;
        this.label = label;
    }

    public int getLabel() {
        return label;
    }

    public float[] getFeatures() {
        return features;
    }

    public Matrix2 toX() {
        return new Matrix2(this.features.length, 1, this.features);
    }

    public Matrix2 toY() {
        return new Matrix2(this.label);
    }

    public static Matrix2 toX(List<SampleItem> items) {
        List<Matrix2> list = new ArrayList<>(items.size());
        for (SampleItem item : items) {
            list.add(item.toX());
        }
        return Matrix2.appendColumns(list);
    }

    public static Matrix2 toY(List<SampleItem> items) {
        List<Matrix2> list = new ArrayList<>(items.size());
        for (SampleItem item : items) {
            list.add(item.toY());
        }
        return Matrix2.appendColumns(list);
    }

    public static Map<Integer, List<SampleItem>> toMap(List<SampleItem> items) {
        Map<Integer, List<SampleItem>> map = new HashMap<>();
        for (SampleItem item : items) {
            List<SampleItem> list = map.get(item.label);
            if(list == null) {
                list = new ArrayList<>();
                map.put(item.label, list);
            }
            list.add(item);
        }
        return map;
    }
}
