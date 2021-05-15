package ml_6002b_coursework;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Debug;
import weka.core.Instance;
import weka.core.Instances;

import java.io.FileReader;
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

    public static int[][] confusionMatrix(int[] predicted, int[] actual) {
        int[][] out = new int[3][3];
        out[0][1] = 0;
        out[1][0] = 0;
        out[0][2] = 1;
        out[2][0] = 1;
        System.out.println(("actual/predicted = y/x"));

        for(int i=0; i < predicted.length; i++) {
            if(predicted[i] == actual[i] && actual[i] == 0) {
                out[1][1]++;
            }
            else if(predicted[i] == actual[i] && actual[i] == 1) {
                out[2][2]++;
            }
            else if(actual[i] == 0) {
                out[2][1]++;
            }
            else if(actual[i] == 1) {
                out[1][2]++;
            }
        }
        return out;
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
