import java.util.Random;

public class Neuron {
    private double[] inputs;
    private double[] weights;
    private double error;

    Neuron(int weightsNumber, int inputNumber) {
        Random random = new Random();

        this.inputs = new double[inputNumber];
        this.weights = new double[weightsNumber];
        for (int i = 0; i < weights.length; ++i) {
            weights[i] = random.nextDouble();
        }
    }

    public void setInput(double[] dataRow) {
        for (int i = 0; i < inputs.length; ++i){
            inputs[i] = dataRow[i];
        }
    }

    public void updateWeights() {
        for (int i = 0; i < inputs.length; ++i)
            weights[i] += error * inputs[i];
    }

    public double getOutput() {
        double activation = 0;

        for (int i = 0; i < inputs.length; ++i) {
            activation += inputs[i]*weights[i];
        }

        return evaluateSigmoidValue(activation);
    }

    public void setError(double error) {
        this.error = error;
    }

    public double getError() {
        return this.error;
    }

    public double getWeight (int i) { return weights[i]; }

    private double evaluateSigmoidValue(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

}
