package deepNN;

import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.Arrays;
import java.util.Collection;
import java.util.Random;

/**
 * A nxm float immutable matrix.
 * Operations are not optimized (for acadamedic purposes only)
 * 
 * @author Matias Leone
 */
public class Matrix2 {
    
    private static final NumberFormat FORMAT = new DecimalFormat("0.###");
    
    private final float[] data;
    private final int rows;
    private final int cols;
    
    /*-------------------------- Creating methods --------------------------*/
    
    public Matrix2(int rows, int cols) {
        if(rows < 1 || cols < 1)
            error("Invalid shape (" + rows + ", " + cols + ")");
        this.rows = rows;
        this.cols = cols;
        this.data = new float[rows * cols];
    }
    
    public Matrix2(int rows, int cols, float[] data) {
        this(rows, cols);
        if(data.length != this.data.length)
            error("Invalid data length: " + data.length);
        this.set(data);
    }
    
    public Matrix2(float value) {
        this(1, 1, new float[]{value});
    }
    
    public Matrix2(float[][] values) {
        this(values.length, values[0].length);
        for (int i = 0; i < values.length; i++) {
            if(values[i].length != this.cols)
                error("Invalid shape: " + Arrays.toString(values));
            this.setRow(i, values[i]);
        }
    }
    
    public static Matrix2 fromValue(int rows, int cols, float v) {
        Matrix2 m = new Matrix2(rows, cols);
        Arrays.fill(m.data, v);
        return m;
    }
    
    public static Matrix2 zeros(int rows, int cols) {
        return fromValue(rows, cols, 0);
    }
    
    public static Matrix2 ones(int rows, int cols) {
        return fromValue(rows, cols, 1);
    }
    
    public static Matrix2 random(int rows, int cols, long randSeed) {
        return new Matrix2(rows, cols).apply(new RandomOp(randSeed));
    }
    
    
    /*-------------------------- Internal methods --------------------------*/
    
    
    private void set(int row, int col, float v) {
        this.data[this.pos(row, col)] = v;
    }
    
    private void set(float[] values) {
        System.arraycopy(values, 0, this.data, 0, values.length);
    }
    
    private void setRow(int row, float[] values) {
        System.arraycopy(values, 0, this.data, rowStart(row), this.cols);
    }
    
    private int pos(int row, int col) {
        return row * this.cols + col;
    }
    
    private int rowStart(int row) {
        return row * this.cols;
    }
    
    private int rowEnd(int row) {
        return rowStart(row) + this.cols;
    }
    
    private Matrix2 emptyCopy() {
        return new Matrix2(this.rows, this.cols);
    }
    
    private static void copyRow(Matrix2 src, int srcRow, Matrix2 dst, int dstRow) {
        System.arraycopy(src.data, src.rowStart(srcRow), dst.data, dst.rowStart(dstRow), src.cols);
    }
    
    private static void copyColumn(Matrix2 src, int srcCol, Matrix2 dst, int dstCol) {
        for (int row = 0; row < src.rows; row++) {
            dst.set(row, dstCol, src.get(row, srcCol));
        }
    }
    
    private static void error(String msg) {
        throw new RuntimeException(msg);
    }
    
    /*-------------------------- Instance methods --------------------------*/
    
    public float get(int row, int col) {
        if(row < 0 || row >= this.rows)
            error("Invalid row: " + row);
        if(col < 0 || col >= this.cols)
            error("Invalid cols: " + col);
        return this.data[pos(row, col)];
    }
    
    public int rows() {
        return this.rows;
    }
    
    public int cols() {
        return this.cols;
    }
    
    public Matrix2 apply(ElementWiseOp op) {
        return Matrix2.apply(this, op);
    }
    
    public Matrix2 mul(float s) {
        return apply(new MulOp(s));
    }
    
    public Matrix2 add(float s) {
        return apply(new AddOp(s));
    }
    
    public Matrix2 sub(float s) {
        return apply(new SubOp(s));
    }
    
    public Matrix2 div(float s) {
        return apply(new DivOp(s));
    }
    
    public Matrix2 scalarMinus(float s) {
        return apply(new ScalarMinusOp(s));
    }
    
    public Matrix2 oneMinus() {
        return apply(ScalarMinusOp.ONE_MINUS);
    }
    
    public Matrix2 log() {
        return apply(LogOp.INSTANCE);
    }
    
    public Matrix2 sigmoid() {
        return apply(SigmoidOp.INSTANCE);
    }
    
    public Matrix2 relu() {
        return apply(ReluOp.INSTANCE);
    }
    
    public Matrix2 greater(float v) {
        return apply(new GreaterOp(v));
    }
    
    public Matrix2 lower(float v) {
        return apply(new LowerOp(v));
    }
    
    public Matrix2 pow(float s) {
        return apply(new PowerOp(s));
    }
    
    public Matrix2 square() {
        return apply(PowerOp.SQ_INSTANCE);
    }
    
    public Matrix2 sqrt() {
        return apply(SqrtOp.INSTANCE);
    }
    
    public Matrix2 mul(Matrix2 m) {
        return Matrix2.mul(this, m);
    }
    
    public Matrix2 add(Matrix2 m) {
        return Matrix2.add(this, m);
    }
    
    public Matrix2 sub(Matrix2 m) {
        return Matrix2.sub(this, m);
    }
    
    public Matrix2 mulEW(Matrix2 m) {
        return Matrix2.mulEW(this, m);
    }
    
    public Matrix2 divEW(Matrix2 m) {
        return Matrix2.divEW(this, m);
    }
    
    public Matrix2 transpose() {
        return Matrix2.transpose(this);
    }
    
    public Matrix2 broadcastCol(int cols) {
        return Matrix2.broadcastCol(this, cols);
    }
    
    public Matrix2 broadcastRow(int rows) {
        return Matrix2.broadcastRow(this, rows);
    }

    public Matrix2 sumColumns() {
        return Matrix2.sumColumns(this);
    }
    
    public Matrix2 sumRows() {
        return Matrix2.sumRows(this);
    }
    
    public float sum() {
        return Matrix2.sum(this);
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder(this.data.length * 2);
        sb.append("Shape(").append(this.rows).append(", ").append(this.cols).append(")\n");
        sb.append("[");
        int maxRows = Math.min(this.rows, 6);
        int maxCols = Math.min(this.cols, 10);
        for (int row = 0; row < maxRows; row++) {
            sb.append("[");
            for (int col = 0; col < maxCols; col++) {
                if(col > 0) {
                    sb.append(", ");
                }
                sb.append(FORMAT.format(get(row, col)));
            }
            if(this.cols > maxCols)
                sb.append(", ...");
            sb.append("]");
            if(row < this.rows - 1)
                sb.append("\n");
        }
        if(this.rows > maxRows)
            sb.append("...");
        sb.append("]");
        
        return sb.toString();
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) {
            return true;
        }
        if (obj == null) {
            return false;
        }
        if (getClass() != obj.getClass()) {
            return false;
        }
        final Matrix2 m = (Matrix2) obj;
        if(m.cols != this.cols || m.rows != this.rows)
            return false;
        
        return Arrays.equals(this.data, m.data);
    }

    
    
    
    /*-------------------------- Static methods --------------------------*/
    
    
    public static Matrix2 broadcastCol(Matrix2 m, int cols) {
        if(m.cols > 1)
            error("Broadcast not supported for more than 1 column");
        if(cols < 1)
            error("Invalid broadcast number: " + cols);
        
        Matrix2 r = new Matrix2(m.rows, cols);
        for (int col = 0; col < cols; col++) {
            Matrix2.copyColumn(m, 0, r, col);
        }
        return r;
    }
    
    public static Matrix2 broadcastRow(Matrix2 m, int rows) {
        if(m.rows == 1)
            error("Broadcast not supported for more than 1 row");
        if(rows < 1)
            error("Invalid broadcast number: " + rows);
        
        Matrix2 r = new Matrix2(rows, m.cols);
        for (int row = 0; row < rows; row++) {
            Matrix2.copyRow(m, 0, r, row);
        }
        return r;
    }
    
    public static Matrix2 sumColumns(Matrix2 m) {
        Matrix2 r = new Matrix2(m.rows, 1);
        for (int row = 0; row < m.rows; row++) {
            float sum = 0;
            for (int col = 0; col < m.cols; col++) {
                sum += m.get(row, col);
            }
            r.set(row, 0, sum);
        }
        return r;
    }
    
    public static Matrix2 sumRows(Matrix2 m) {
        Matrix2 r = new Matrix2(1, m.cols);
        for (int col = 0; col < m.cols; col++) {
            float sum = 0;
            for (int row = 0; row < m.rows; row++) {
                sum += m.get(row, col);
            }
            r.set(0, col, sum);
        }
        return r;
    }
    
    public static float sum(Matrix2 m) {
        float sum = 0;
        for (int row = 0; row < m.rows; row++) {
            for (int col = 0; col < m.cols; col++) {
                sum += m.get(row, col);
            }
        }
        return sum;
    }
    
    public static Matrix2 apply(Matrix2 m, ElementWiseOp op) {
        Matrix2 r = m.emptyCopy();
        for (int row = 0; row < m.rows; row++) {
            for (int col = 0; col < m.cols; col++) {
                r.set(row, col, op.apply(m.get(row, col)));
            }
        }
        return r;
    }
    
    public static Matrix2 apply(Matrix2 a, Matrix2 b, ElementWise2MatOp op) {
        if(!sameShape(a, b))
            error("Invalid shapes, a: " + a + ", b: " + b);
        
        Matrix2 r = a.emptyCopy();
        for (int row = 0; row < a.rows; row++) {
            for (int col = 0; col < a.cols; col++) {
                r.set(row, col, op.apply(a.get(row, col), b.get(row, col)));
            }
        }
        return r;
    }
    
    public static Matrix2 mul(Matrix2 a, Matrix2 b) {
        if(a.cols != b.rows)
            error("Invalid shapes, a: " + a + ", b: " + b);
        
        Matrix2 c = new Matrix2(a.rows, b.cols);
        for (int row = 0; row < a.rows; row++) {
            for (int col = 0; col < b.cols; col++) {
                c.data[c.pos(row, col)] = rowColumnDot(a, row, b, col);
            }
        }
        return c;
    }
    
    private static float rowColumnDot(Matrix2 a, int row, Matrix2 b, int col) {
        float dot = 0;
        for (int i = 0; i < a.cols; i++) {
            dot += a.get(row, i) * b.get(i, col);
        }
        return dot;
    }
    
    public static Matrix2 add(Matrix2 a, Matrix2 b) {
        return Matrix2.apply(a, b, AddMatOp.INSTANCE);
    }
    
    public static Matrix2 sub(Matrix2 a, Matrix2 b) {
        return Matrix2.apply(a, b, SubMatOp.INSTANCE);
    }
    
    public static boolean sameShape(Matrix2 a, Matrix2 b) {
        return a.rows == b.rows && a.cols == b.cols;
    }
    
    public static Matrix2 mulEW(Matrix2 a, Matrix2 b) {
        return Matrix2.apply(a, b, MulMatEWOp.INSTANCE);
    }
    
    public static Matrix2 divEW(Matrix2 a, Matrix2 b) {
        return Matrix2.apply(a, b, DivMatEWOp.INSTANCE);
    }
    
    public static Matrix2 transpose(Matrix2 m) {
        Matrix2 t = new Matrix2(m.cols, m.rows);
        for (int row = 0; row < m.rows; row++) {
            for (int col = 0; col < m.cols; col++) {
                t.set(col, row, m.get(row, col));
            }
        }
        return t;
    }
    
    public static Matrix2 appendColumns(Collection<Matrix2> list) {
        int rows = 0;
        int cols = 0;
        for (Matrix2 m : list) {
            if(rows == 0) {
                rows = m.rows;
            } else {
                if(m.rows != rows)
                    error("Invalid number of rows in: " + m);
            }
            cols += m.cols;
        }
        
        Matrix2 r = new Matrix2(rows, cols);
        int colIdx = 0;
        for (Matrix2 m : list) {
            for (int col = 0; col < m.cols; col++) {
                Matrix2.copyColumn(m, col, r, colIdx);
                colIdx++;
            }
        }
        return r;
    }
    
    public static Matrix2 getColumns(Matrix2 m, int[] indices) {
        if(indices == null || indices.length == 0)
            error("Invalid indices: " + Arrays.toString(indices));
        
        Matrix2 r = new Matrix2(m.rows, indices.length);
        for (int i = 0; i < indices.length; i++) {
            int col = indices[i];
            if(col < 0 || col >= m.cols)
                error("Invalid column index: " + col);
            Matrix2.copyColumn(m, col, r, i);
        }
        return r;
    }

    public static Matrix2 getRows(Matrix2 m, int[] indices) {
        if(indices == null || indices.length == 0)
            error("Invalid indices: " + Arrays.toString(indices));
        
        Matrix2 r = new Matrix2(indices.length, m.cols);
        for (int i = 0; i < indices.length; i++) {
            int row = indices[i];
            if(row < 0 || row >= m.cols)
                error("Invalid row index: " + row);
            Matrix2.copyRow(m, row, r, i);
        }
        return r;
    }
    
    
    
    
    
    /*-------------------------- Element-wise methods --------------------------*/
    
    
    /**
     * Element wise operation
     */
    public interface ElementWiseOp {
        float apply(float v);
    }
    
    public static class RandomOp implements ElementWiseOp {
        private final Random rand;
        public RandomOp(long randSeed) {
            this.rand = new Random(randSeed);
        }
        @Override
        public float apply(float v) {
            return (float)rand.nextGaussian();
        }
    }
    
    public static abstract class ScalarOp implements ElementWiseOp {
        protected final float s;
        public ScalarOp(float s) {
            this.s = s;
        }
    }
    
    public static class MulOp extends ScalarOp {
        public MulOp(float s) {
            super(s);
        }
        @Override
        public float apply(float v) {
            return v * s;
        }
    }
    
    public static class AddOp extends ScalarOp {
        public AddOp(float s) {
            super(s);
        }
        @Override
        public float apply(float v) {
            return v + s;
        }
    }
    
    public static class SubOp extends ScalarOp {
        public SubOp(float s) {
            super(s);
        }
        @Override
        public float apply(float v) {
            return v - s;
        }
    }
    
    public static class DivOp extends ScalarOp {
        public DivOp(float s) {
            super(s);
        }
        @Override
        public float apply(float v) {
            return v / s;
        }
    }
    
    public static class GreaterOp extends ScalarOp {
        public GreaterOp(float s) {
            super(s);
        }
        @Override
        public float apply(float v) {
            return v > s ? 1f : 0f;
        }
    }
    
    public static class LowerOp extends ScalarOp {
        public LowerOp(float s) {
            super(s);
        }
        @Override
        public float apply(float v) {
            return v < s ? 1f : 0f;
        }
    }
    
    public static class PowerOp extends ScalarOp {
        public static final ElementWiseOp SQ_INSTANCE = new PowerOp(2);
        public PowerOp(float s) {
            super(s);
        }
        @Override
        public float apply(float v) {
            return (float)Math.pow(v, s);
        }
    }
    
    public static class SqrtOp implements ElementWiseOp {
        public static final ElementWiseOp INSTANCE = new SqrtOp();
        @Override
        public float apply(float v) {
            return (float)Math.sqrt(v);
        }
        
    }
    
    public static class ScalarMinusOp extends ScalarOp {
        public static final ElementWiseOp ONE_MINUS = new ScalarMinusOp(1);
        public ScalarMinusOp(float s) {
            super(s);
        }
        @Override
        public float apply(float v) {
            return s - v;
        }
    }
    
    public static class SigmoidOp implements ElementWiseOp {
        public static final ElementWiseOp INSTANCE = new SigmoidOp();
        @Override
        public float apply(float v) {
            return 1f / (1f + (float)Math.exp(-v));
        }
    }
    
    public static class ReluOp implements ElementWiseOp {
        public static final ElementWiseOp INSTANCE = new ReluOp();
        @Override
        public float apply(float v) {
            return Math.max(0, v);
        }
    }
    
    public static class LogOp implements ElementWiseOp {
        public static final ElementWiseOp INSTANCE = new LogOp();
        @Override
        public float apply(float v) {
            return (float)Math.log(v);
        }
    }
    
    /**
     * Element wise operation between two matrix
     */
    public interface ElementWise2MatOp {
        float apply(float a, float b);
    }
    
    public static class AddMatOp implements ElementWise2MatOp {
        public static final ElementWise2MatOp INSTANCE = new AddMatOp();
        @Override
        public float apply(float a, float b) {
            return a + b;
        }
    }
    
    public static class SubMatOp implements ElementWise2MatOp {
        public static final ElementWise2MatOp INSTANCE = new SubMatOp();
        @Override
        public float apply(float a, float b) {
            return a - b;
        }
    }
    
    public static class MulMatEWOp implements ElementWise2MatOp {
        public static final ElementWise2MatOp INSTANCE = new MulMatEWOp();
        @Override
        public float apply(float a, float b) {
            return a * b;
        }
    }
    
    public static class DivMatEWOp implements ElementWise2MatOp {
        public static final ElementWise2MatOp INSTANCE = new DivMatEWOp();
        @Override
        public float apply(float a, float b) {
            return a / b;
        }
    }
    
}
