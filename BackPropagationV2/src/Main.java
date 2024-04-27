import java.util.Arrays;

public class Main {
    static int TABLESIZE = 30;
    static int TRAINTABLESIZE = 27;

    public static void main(String[] args) {
        double[][] table = createInputTable();
        normalizeInputTable(table);
        int iterationsNumber = 50000;

        BackPropagation algo = new BackPropagation(iterationsNumber, Arrays.copyOfRange(table, 0, TRAINTABLESIZE - 1));
        algo.startTraining();
        double avgError = 0;
        for (int i = TRAINTABLESIZE; i < TABLESIZE; ++i) {
            algo.evaluate(table[i]);
            avgError = getError(algo.getEvaluatedValue(), table[i][3]);
            System.out.printf("%7s | %7s | %7s\n","Answer", "Table", "Error");
            System.out.printf("%.5f | %.5f | %.5f\n", algo.getEvaluatedValue(), table[i][3], getError(algo.getEvaluatedValue(), table[i][3]));
        }
        avgError /= 3;
        System.out.printf("\nAverage error: " + avgError);

    }
    private static double getError(double x, double y){
        return Math.abs(x-y);
    }
    private static double[][] createInputTable() {
        double averageValue = 0;
        double[][] inputTable = new double[TABLESIZE][5];

        for (int i = 0; i < inputTable.length; ++i) {
            int x, y, z;
            switch (i / 10) {
                case 0:
                    x = 1; y = 2; z = i + 1; break;
                case 1:
                    x = 1; y = i - 9; z = 3; break;
                default:
                    x = i - 19; y = 2; z = 3; break;
            }
            inputTable[i][0] = x;
            inputTable[i][1] = y;
            inputTable[i][2] = z;

            double functionValue = getFunctionValue(x, y, z);
            inputTable[i][3] = functionValue;
            if (i < TRAINTABLESIZE) averageValue += functionValue;
        }

        averageValue /= TRAINTABLESIZE;
        for (double[] row : inputTable)
            row[4] = (row[3] > averageValue) ? 1 : 0;

        return inputTable;
    }

    private static void normalizeInputTable(double[][] inputTable) {
        for (int j = 0; j < inputTable[0].length; j++) {
            double max = inputTable[0][j];
            double min = inputTable[0][j];

            for (double[] trainInput : inputTable) {
                max = Math.max(max, trainInput[j]);
                min = Math.min(min, trainInput[j]);
            }

            for (double[] row : inputTable)
                row[j] = (max - row[j]) / (max - min);
        }
    }

    private static double getFunctionValue(int x, int y, int z) {
        return (x*x-y*y+z*z);
    }
}