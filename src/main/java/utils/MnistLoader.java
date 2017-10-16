package utils;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

public class MnistLoader {

    public static List<SampleItem> loadMnistData(String xPath, String yPath) {
        List<SampleItem> list = new ArrayList<>();

        InputStream inputX = MnistLoader.class.getClassLoader().getResourceAsStream(xPath);
        InputStream inputY = MnistLoader.class.getClassLoader().getResourceAsStream(yPath);
        if(inputX == null || inputY == null) {
            return list;
        }

        BufferedReader rx = null;
        BufferedReader ry = null;
        try {
            rx = new BufferedReader(new InputStreamReader(inputX));
            ry = new BufferedReader(new InputStreamReader(inputY));
            String lineX, lineY;
            while((lineX = rx.readLine()) != null) {
                lineX = lineX.trim();
                if(lineX.isEmpty())
                    continue;
                lineY = ry.readLine();

                String[] values = lineX.split(",");
                int label = Integer.parseInt(lineY.trim());
                if(label == 10) {
                    label = 0;
                }
                float[] data = new float[values.length];
                for (int i = 0; i < values.length; i++) {
                    data[i] = Float.parseFloat(values[i]);
                }
                list.add(new SampleItem(data, label));
            }

            return list;

        } catch (Exception e) {
            throw new RuntimeException("Error reading MNist data from files: " + xPath + " and " + yPath, e);
        } finally {
            if(rx != null) {
                try {
                    rx.close();
                } catch (Exception e) {
                }
            }
            if(ry != null) {
                try {
                    ry.close();
                } catch (Exception e) {
                }
            }
        }
    }
}
