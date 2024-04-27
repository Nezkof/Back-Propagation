import com.sun.source.tree.NewArrayTree;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

public class BackPropagation {
    private final int FLNN = 4;
    private final int SLNN = 2;

    private final int iterationsNumber;
    private final double[][] data;
    private final Neuron[] firstLayer;
    private final Neuron[] secondLayer;

    public BackPropagation(int iterationsNumber, double[][] data) {
        this.iterationsNumber = iterationsNumber;


        this.firstLayer = new Neuron[FLNN];
        for (int i = 0; i < firstLayer.length; ++i) {
            firstLayer[i] = new Neuron(FLNN, 3);
        }

        this.secondLayer = new Neuron[SLNN];

        for (int i = 0; i < secondLayer.length; ++i) {
            secondLayer[i] = new Neuron(FLNN, FLNN);
        }

        this.data = data;
    }

    public void startTraining() {
        for (int i = 0; i < iterationsNumber; ++i) {
            List<double[]> dataList = Arrays.asList(data);
            Collections.shuffle(dataList);
            dataList.toArray(data);

            for (double[] dataRow : data) {
                double[] firstLayerOutputs = new double[FLNN];

                for (var neuron : firstLayer)
                    neuron.setInput(Arrays.copyOfRange(dataRow, 0, 3));

                for (var neuron : secondLayer) {
                    for (int k = 0; k < firstLayerOutputs.length; ++k)
                        firstLayerOutputs[k] = firstLayer[k].getOutput();
                    neuron.setInput(firstLayerOutputs);
                }

                for (var neuron : secondLayer)
                    neuron.setError(getDerivativeValue(secondLayer[0].getOutput()) * (dataRow[3] - secondLayer[0].getOutput()));;

                for (int j = 0; j < firstLayer.length; ++j)
                    firstLayer[j].setError((secondLayer[0].getError() * secondLayer[0].getWeight(j) + secondLayer[1].getError() * secondLayer[1].getWeight(j)) * getDerivativeValue(firstLayer[j].getOutput()));

                for (var firstLevelNeuron : firstLayer)
                    firstLevelNeuron.updateWeights();

                for (var secondLeverNeuron : secondLayer)
                    secondLeverNeuron.updateWeights();
            }
        }

    }

    public void evaluate(double[] testingData) {
        double[] firstLayerOutputs = new double[FLNN];
        for (var firstLeverNeuron : firstLayer)
            firstLeverNeuron.setInput(Arrays.copyOfRange(testingData, 0, 3));

        for (var neuron : secondLayer) {
            for (int k = 0; k < firstLayerOutputs.length; ++k)
                firstLayerOutputs[k] = firstLayer[k].getOutput();
            neuron.setInput(firstLayerOutputs);
        }
    }

    public double getEvaluatedValue() {
        return secondLayer[0].getOutput();
    }
    private double getDerivativeValue(double x) { return x * (1 - x); }


}
