import java.util.Random;

public class Neuron {
    private double[] inputs;
    private double[] weights = new double[4];
    private double error;

    public double output() {
        return getSigmoidValue(weights[0] * inputs[0] + weights[1] * inputs[1] + weights[2] * inputs[2]);
    }

    public void setRandomWeights() {
        Random random = new Random();
        for (int i = 0; i < weights.length; ++i) {
            weights[i] = random.nextDouble();
        }
    }

    public void updateWeights() {
        for (int i = 0; i < inputs.length; ++i)
            weights[i] += error * inputs[i];
    }

    public double getWeight (int i) { return weights[i]; }
    public double getError () { return error; }
    public double getSigmoidValue(double x) { return 1.0 / (1.0 + Math.exp(-x)); }

    public void setInputs(double input1, double input2) { inputs = new double[] { input1, input2}; }
    public void setInputs(double input1, double input2, double input3) { inputs = new double[] { input1, input2, input3 }; }
    public void setInputs(double input1, double input2, double input3, double input4) { inputs = new double[] { input1, input2, input3, input4 }; }
    public void setError(double error){ this.error = error; }
}