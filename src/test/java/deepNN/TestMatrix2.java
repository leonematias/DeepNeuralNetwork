package deepNN;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

/**
 * Unit tests for Matrix2 class
 *
 * @author matias.leone
 */
public class TestMatrix2 {
    
    private static final float EPSILON = 0.0001f;
    
    
    public TestMatrix2() {
    }
    
    @Before
    public void setUp() {
    }
    
    @After
    public void tearDown() {
    }
    
    @Test
    public void testShape() {
        Matrix2 a = new Matrix2(new float[][]{
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9},
            {10, 11, 12}
        });
        
        assertEquals(4, a.rows());
        assertEquals(3, a.cols());
        assertEquals(9, a.get(2, 2), EPSILON);
        assertEquals(10, a.get(3, 0), EPSILON);
    }
    
    @Test
    public void testMulScalar() {
        Matrix2 a = new Matrix2(new float[][]{
            {1, 2, 3},
            {4, 5, 6},
        });
        Matrix2 res = a.mul(2);
        Matrix2 expected = new Matrix2(new float[][]{
            {2, 4, 6},
            {8, 10, 12},
        });
        assertEquals(expected, res);
    }
    
    @Test
    public void testAddScalar() {
        Matrix2 a = new Matrix2(new float[][]{
            {1, 2, 3},
            {4, 5, 6},
        });
        Matrix2 res = a.add(10);
        Matrix2 expected = new Matrix2(new float[][]{
            {11, 12, 13},
            {14, 15, 16},
        });
        assertEquals(expected, res);
    }
    
    @Test
    public void testSubScalar() {
        Matrix2 a = new Matrix2(new float[][]{
            {1, 2, 3},
            {4, 5, 6},
        });
        Matrix2 res = a.sub(10);
        Matrix2 expected = new Matrix2(new float[][]{
            {1-10, 2-10, 3-10},
            {4-10, 5-10, 6-10},
        });
        assertEquals(expected, res);
    }
    
    @Test
    public void testAddMatrix() {
        Matrix2 a = new Matrix2(new float[][]{
            {1, 2, 3},
            {4, 5, 6},
        });
        Matrix2 res = a.add(a);
        Matrix2 expected = new Matrix2(new float[][]{
            {2, 4, 6},
            {8, 10, 12},
        });
        assertEquals(expected, res);
    }
    
    @Test
    public void testSubMatrix() {
        Matrix2 a = new Matrix2(new float[][]{
            {1, 2, 3},
            {4, 5, 6},
        });
        Matrix2 res = a.sub(a);
        Matrix2 expected = Matrix2.zeros(a.rows(), a.cols());
        assertEquals(expected, res);
    }
    
    @Test
    public void testMatrixMul() {
        Matrix2 a = new Matrix2(new float[][]{
            {1, 2, 3},
            {4, 5, 6}
        });
        Matrix2 b = new Matrix2(new float[][]{
                {7, 8},
                {9, 10},
                {11, 12},
            });
        Matrix2 expected = new Matrix2(new float[][]{
                {58, 64},
                {139, 154}
            });
        Matrix2 c = Matrix2.mul(a, b);
        assertEquals(expected, c);
    }
    
    @Test
    public void testMatrixMulElementWise() {
        Matrix2 a = new Matrix2(new float[][]{
            {1, 2, 3},
            {4, 5, 6}
        });
        Matrix2 expected = new Matrix2(new float[][]{
                {1, 4, 9},
                {16, 25, 36}
            });
        Matrix2 res = a.mulEW(a);
        assertEquals(expected, res);
    }
    
    @Test
    public void testMatrixDivElementWise() {
        Matrix2 a = new Matrix2(new float[][]{
            {1, 2, 3},
            {4, 5, 6}
        });
        Matrix2 expected = Matrix2.ones(a.rows(), a.cols());
        Matrix2 res = a.divEW(a);
        assertEquals(expected, res);
    }
    
    @Test
    public void testBrodcastCol() {
        Matrix2 a = new Matrix2(new float[][]{
            {1},
            {1},
            {1},
        });
        Matrix2 expected = new Matrix2(new float[][]{
                {1, 1, 1},
                {1, 1, 1},
                {1, 1, 1}
            });
        Matrix2 c = a.broadcastCol(3);
        assertEquals(expected, c);
    }
    
    @Test
    public void testSumColumns() {
        Matrix2 a = new Matrix2(new float[][]{
            {1, 2, 3},
            {4, 5, 6},
            {7, 8, 9},
            {10, 11, 12}
        });
        Matrix2 res = a.sumColumns();
        assertEquals(4, res.rows());
        assertEquals(1, res.cols());
        assertEquals(1+2+3, res.get(0, 0), EPSILON);
        assertEquals(4+5+6, res.get(1, 0), EPSILON);
        assertEquals(7+8+9, res.get(2, 0), EPSILON);
        assertEquals(10+11+12, res.get(3, 0), EPSILON);
    }
    
    @Test
    public void testTranspose() {
        Matrix2 a = new Matrix2(new float[][]{
            {1, 2, 3},
            {4, 5, 6}
        });
        Matrix2 expected = new Matrix2(new float[][]{
                {1, 4},
                {2, 5},
                {3, 6}
            });
        Matrix2 res = a.transpose();
        assertEquals(expected, res);
    }
    
    @Test
    public void testGreater() {
        Matrix2 a = new Matrix2(new float[][]{
            {1, 2, 3},
            {4, 5, 6}
        });
        Matrix2 expected = new Matrix2(new float[][]{
            {0, 0, 0},
            {0, 1, 1}
        });
        Matrix2 res = a.greater(4);
        assertEquals(expected, res);
    }
    
    @Test
    public void testLower() {
        Matrix2 a = new Matrix2(new float[][]{
            {1, 2, 3},
            {4, 5, 6}
        });
        Matrix2 expected = new Matrix2(new float[][]{
            {1, 1, 1},
            {0, 0, 0}
        });
        Matrix2 res = a.lower(4);
        assertEquals(expected, res);
    }
    
    @Test
    public void testOneMinus() {
        Matrix2 a = new Matrix2(new float[][]{
            {1, 2, 3},
            {4, 5, 6}
        });
        Matrix2 expected = new Matrix2(new float[][]{
            {1-1, 1-2, 1-3},
            {1-4, 1-5, 1-6}
        });
        Matrix2 res = a.oneMinus();
        assertEquals(expected, res);
    }
    
    @Test
    public void testLog() {
        Matrix2 a = Matrix2.fromValue(1, 1, 10);
        Matrix2 expected = Matrix2.fromValue(1, 1, (float)Math.log(10));
        Matrix2 res = a.log();
        assertEquals(expected, res);
    }
    
    @Test
    public void testSigmoid() {
        Matrix2 a = Matrix2.fromValue(1, 1, 10);
        Matrix2 expected = Matrix2.fromValue(1, 1, 1 / (1 + (float)Math.exp(-10)));
        Matrix2 res = a.sigmoid();
        assertEquals(expected, res);
    }
    
    @Test
    public void testRelu() {
        Matrix2 a = new Matrix2(new float[][]{
            {-30, -10, -1},
            {1, 5, 30}
        });
        Matrix2 expected = new Matrix2(new float[][]{
            {0, 0, 0},
            {1, 5, 30}
        });
        Matrix2 res = a.relu();
        assertEquals(expected, res);
    }
    
}
