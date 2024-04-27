public class Main {

    public static void main(String[] args) {
        int tableSize = 30;
        int trainTableSize = 27;
        int iterationsNumber = 10000;

        BackPropagation algorithm = new BackPropagation(tableSize, trainTableSize, iterationsNumber);
        algorithm.startTraining();
        algorithm.startTesting();
    }
}

