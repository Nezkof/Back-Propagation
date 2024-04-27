import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class BackPropagation {
    private final int tableSize;
    private final int trainTableSize;
    private final int iterationsNumber;
    private int iterations;
    private long testingDuration;
    private final double[][] inputTable;
    private final Neuron[] firstLevelNeurons;
    private final Neuron[] secondLevelNeurons;

    public BackPropagation (int tableSize, int trainTableSize, int iterationsNumber) {
        this.tableSize = tableSize;
        this.trainTableSize = trainTableSize;
        this.iterationsNumber = iterationsNumber;

        inputTable = fillInputTable();
        firstLevelNeurons = new Neuron[] { new Neuron(), new Neuron(), new Neuron(), new Neuron()};
        secondLevelNeurons = new Neuron[]{ new Neuron(), new Neuron()};
    }

    public void startTraining() {
        long startTime = System.currentTimeMillis();

        for (var neuron: firstLevelNeurons)
            neuron.setRandomWeights();

        for (var neuron : secondLevelNeurons)
            neuron.setRandomWeights();

        System.out.println("============================================================================");
        System.out.println("\t\t\t\t\t\tTesting before training");
        System.out.println("----------------------------------------------------------------------------");
        System.out.printf("%4s %7s %7s %10s %7s %15s %10s%n", "x1", "x2", "x3", "d1", "d2", "d1(table)", "d2(table)");
        for (int i = trainTableSize; i < inputTable.length; i++) {
            for (int j = 0; j < firstLevelNeurons.length; ++j)
                firstLevelNeurons[j].setInputs(inputTable[j][0], inputTable[j][1], inputTable[j][2]);

            for (Neuron secondLeverNeuron : secondLevelNeurons)
                secondLeverNeuron.setInputs(firstLevelNeurons[0].output(), firstLevelNeurons[1].output(), firstLevelNeurons[2].output(), firstLevelNeurons[3].output());

        System.out.printf("%.4f  %.4f  %.4f  %9.4f  %.4f  %10.4f  %9.4f%n", inputTable[i][0], inputTable[i][1], inputTable[i][2], secondLevelNeurons[0].output(), secondLevelNeurons[1].output(),inputTable[i][3], inputTable[i][4]);
        }

        for( ; iterations < iterationsNumber; ++iterations) {
            for (int i = 0; i < trainTableSize; i++) {
                for (var firstLevelNeuron : firstLevelNeurons)
                    firstLevelNeuron.setInputs(inputTable[i][0], inputTable[i][1], inputTable[i][2]);

                for (var secondLevelNeuron : secondLevelNeurons )
                    secondLevelNeuron.setInputs(firstLevelNeurons[0].output(),firstLevelNeurons[1].output(), firstLevelNeurons[2].output(), firstLevelNeurons[3].output());

                for (var secondLevelNeuron : secondLevelNeurons)
                    secondLevelNeuron.setError(getDerivativeValue(secondLevelNeurons[0].output()) * (inputTable[i][3] - secondLevelNeurons[0].output()));

                for (int j = 0; j < firstLevelNeurons.length; ++j)
                    firstLevelNeurons[j].setError((secondLevelNeurons[0].getError() * secondLevelNeurons[0].getWeight(j) + secondLevelNeurons[1].getError() * secondLevelNeurons[1].getWeight(j)) * getDerivativeValue(firstLevelNeurons[j].output()));

                for (var firstLevelNeuron : firstLevelNeurons)
                    firstLevelNeuron.updateWeights();

                for (var secondLeverNeuron : secondLevelNeurons)
                    secondLeverNeuron.updateWeights();
            }
        }

        testingDuration = System.currentTimeMillis() - startTime;
    }

    public void startTesting(){
        double qualityd1 = 0;
        double qualityd2 = 0;

        System.out.println("============================================================================");
        System.out.println("\t\t\tTesting after " + iterations + " iterations and " + testingDuration + " milliseconds");
        System.out.println("----------------------------------------------------------------------------");
        System.out.printf("%4s %7s %7s %10s %7s %15s %10s%n", "x1", "x2", "x3", "d1", "d2", "d1(table)", "d2(table)");
        for (int i = trainTableSize; i < inputTable.length; i++) {
            for (var firstLeverNeuron : firstLevelNeurons)
                firstLeverNeuron.setInputs(inputTable[i][0], inputTable[i][1], inputTable[i][2]);

            for (var secondLeverNeuron : secondLevelNeurons)
                secondLeverNeuron.setInputs(firstLevelNeurons[0].output(), firstLevelNeurons[1].output(), firstLevelNeurons[2].output(), firstLevelNeurons[3].output());

        System.out.printf("%.4f  %.4f  %.4f  %9.4f  %.4f  %10.4f  %9.4f%n", inputTable[i][0], inputTable[i][1], inputTable[i][2], secondLevelNeurons[0].output(), secondLevelNeurons[1].output(),inputTable[i][3], inputTable[i][4]);
        qualityd1 += secondLevelNeurons[0].output() - inputTable[i][3];
        qualityd2 += secondLevelNeurons[1].output() - inputTable[i][4];
        }
        qualityd1 /= tableSize - trainTableSize;
        qualityd2 /= tableSize - trainTableSize;
        System.out.println("Accuracy for d1:" + qualityd1);
        System.out.println("Accuracy for d2:" + qualityd2);
    }

    private double[][] fillInputTable(){
        double averageValue = 0;
        double[][] inputTable = new double[tableSize][5];

        for (int i = 0; i < inputTable.length; ++i){
            if (i < 10) {
                inputTable[i][0] = 1;
                inputTable[i][1] = 2;
                inputTable[i][2] = i+1;
            }
            else if (i < 20) {
                inputTable[i][0] = 1;
                inputTable[i][1] = i-9;
                inputTable[i][2] = 3;
            }
            else {
                inputTable[i][0] = i-19;
                inputTable[i][1] = 2;
                inputTable[i][2] = 3;
            }

            inputTable[i][3] = getFunctionValue((int)inputTable[i][0], (int)inputTable[i][1], (int)inputTable[i][2]);
            if (i < trainTableSize) averageValue += inputTable[i][3];
        }

        averageValue = averageValue / inputTable.length;
        for (double[] row : inputTable)
            row[4] = (row[3] > averageValue) ? 1 : 0;

        for (int j = 0; j < inputTable[0].length; j++) {
            double max = inputTable[0][j];
            double min = inputTable[0][j];

            for (double[] trainInput : inputTable) {
                max = Math.max(max, trainInput[j]);
                min = Math.min(min, trainInput[j]);
            }

            for (int i = 0; i < inputTable.length; i++)
                inputTable[i][j] = (max - inputTable[i][j]) / (max - min);
        }

/*        for (var row : inputTable)
            System.out.println(Arrays.toString(row));*/

        List<Integer> indexes = new ArrayList<>();
        for (int i = 0; i < 30; i++)
            indexes.add(i);


        Collections.shuffle(indexes);

        double[][] shuffledTable = new double[inputTable.length][inputTable[0].length];
        for (int i = 0; i < inputTable.length; i++) {
            shuffledTable[i] = inputTable[indexes.get(i)];
        }

        return shuffledTable;
    }

    private double getDerivativeValue(double x) { return x * (1 - x); }
    private double getFunctionValue(int x, int y, int z) {
        return (x*x-y*y+z*z);
    }
}
