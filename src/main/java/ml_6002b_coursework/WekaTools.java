package ml_6002b_coursework;

import fileIO.OutFile;
import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Debug;
import weka.core.Instance;
import weka.core.Instances;

import java.io.FileReader;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.Random;

public class WekaTools {

    public static double accuracy(Classifier c, Instances test) throws Exception {
        double count = 0;
        double correct = 0;

        for(Instance instance : test){
            double result = c.classifyInstance(instance);
            double actual = instance.classValue();

            if(result == actual){
                correct++;
            }
            count++;
        }
        //The accuracy of the classifier is then the number correct divided by the number of instances
        double accuracy = ((correct/count)*100);
        return accuracy;
    }

    public static Instances loadClassificationData(String path) {
        Instances train = null;
        try{
            FileReader reader = new FileReader(path);
            train = new Instances(reader);
            train.setClassIndex(train.numAttributes() - 1);
        }catch(Exception e){
            System.out.println("Exception caught:" + e);
        }
        return train;
    }

    public static Instances[] splitData(Instances all, double proportion) {
        Instances[] split = new Instances[2];
        Random random = new Random();
        all.randomize(random);

        int trainSize = (int) Math.round(all.numInstances()*proportion);

        split[0] = new Instances (all);
        split[1] = new Instances (all, 0);
        for(int i = 0; i < trainSize; i++) {
            split[1].add(split[0].firstInstance());
            split[0].remove(split[0].firstInstance());
        }
        //System.out.println(split[0]);
        //System.out.println(split[1]);
        return split;
    }

    public static double[] classDistribution(Instances data) {
        int i = data.numAttributes()-1;
        int[] total = new int[data.attribute(i).numValues()];
        for(Instance instance: data){
            total[(int)instance.value(i)]++;
        }

        double[] dis = new double[total.length];
        for(int j = 0; j < total.length; j++) {
            dis[j]  = (double)total[j] / data.numInstances();
        }
        return dis;
    }

    public static double[][] confusionMatrix(double[] predicted, double[] actual, int numclasses) {
        double[][] out = new double[numclasses][numclasses];
        for(int i=1; i<numclasses; i++) {
            out[0][i] = (double) (i - 1);
            out[i][0] = (double) (i - 1);
        }
        System.out.println(("actual/predicted = y/x"));

        for(int i=0; i < predicted.length; i++) {
            out[(int)actual[i]][(int)predicted[i]]++;
        }
        return out;
    }

    public static double balancedAccuracy(double[][] confusionMatrix) {
        double totalTR = 0;
        for(int i=0; i<confusionMatrix[0].length; i++) {
            double sum = 0;
            for(int j=0; j<confusionMatrix.length; j++) {
                sum = sum + confusionMatrix[i][j];
            }
            totalTR = totalTR + (confusionMatrix[i][i] / sum);
        }
        totalTR = totalTR / confusionMatrix.length;
        return totalTR * 100;
    }

    public static void generateTestResults(Classifier classifier, Instances train, Instances test, String parameter, String outputPath, String outputFile) throws Exception {
        classifier.buildClassifier(train);

        // setup output file
        OutFile out = new OutFile(outputPath + outputFile + ".csv");
        out.writeLine(train.relationName() + "," + classifier.getClass().getSimpleName());
        out.writeLine(parameter);
        out.writeLine(String.valueOf(WekaTools.accuracy(classifier, test)));

        // for each instance in test
        for (Instance ins : test) {
            // get predicted class and probabilities of each class
            int prediction = (int) classifier.classifyInstance(ins);
            double[] probabilities = classifier.distributionForInstance(ins);

            // write actual class and predicted class
            out.writeString((int) ins.classValue() + "," + prediction + ",,");

            // write probabilities of each class
            StringBuilder line = new StringBuilder();
            for (double d : probabilities) {
                line.append(d).append(",");
            }
            line.deleteCharAt(line.length() - 1); // remove tailing ','
            out.writeLine(line.toString());
        }
    }

    public static int[] classifyInstances(Classifier c, Instances test) {
        int[] out = new int[test.numInstances()];
        int count = 0;
        try {
            for(Instance ins:test) {
                out[count] = (int)c.classifyInstance(ins);
                count++;
            }
        }
        catch(Exception e) {
            System.out.println("Exception caught:" + e);
        }
        return out;
    }

    public static int[] getClassValues(Instances data) {
        int[] out = new int[data.numInstances()];
        int count = 0;

        for(Instance ins:data) {
            out[count] = (int)ins.classValue();
            count++;
        }
        return out;
    }
}
