package ml_6002b_coursework;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.util.Arrays;
import java.util.Enumeration;

public class GiniAttributeSplitMeasure implements AttributeSplitMeasure {
    public static void main(String[] args) throws Exception{
        //load in dataset
        Instances data = WekaTools.loadClassificationData("src/main/java/ml_6002b_coursework/test_data/meningitis.arff");
        //set target attribute
        data.setClassIndex(data.numAttributes()-1);
        //compute attribute quality for each attribute on base dataset and print to usesr
        Enumeration attEnum = data.enumerateAttributes();
        double attQuality;
        while (attEnum.hasMoreElements()) {
            Attribute att = (Attribute) attEnum.nextElement();
            attQuality = new GiniAttributeSplitMeasure().computeAttributeQuality(data, att);
            System.out.println("Measure Gini index for Attribute " + att.name() + " splitting diagnosis: " + attQuality);
        }
        /*
        try {
            ID3Coursework c = new ID3Coursework();
            String[] options = {"-S", "g"};
            c.setOptions(options);
            c.buildClassifier(data);
        }
        catch(Exception e) {
            System.out.println("Exception: " + e);
        }*/
    }

    @Override
    public double computeAttributeQuality(Instances data, Attribute att) throws Exception {
        // Gini at root node
        double gini = calculateGini(data);

        Instances[] splitData;
        int numValues;

        if (att.isNominal()) {
            splitData = splitData(data, att);
            numValues = att.numValues();
        }
        else {
            splitData = splitDataOnNumeric(data, att).getKey();
            numValues = splitData.length;
        }

        for (int j = 0; j < splitData.length; j++) {
            if (splitData[j].numInstances() > 0) {
                gini -= ((double)splitData[j].numInstances() /
                        (double)data.numInstances()) *
                        calculateGini(splitData[j]);
            }
        }
        return gini;
    }

    private double calculateGini(Instances data) {
        double [] classCounts = new double[data.numClasses()];
        double impurity = 1.0;
        double numInstances = data.numInstances();

        Enumeration instEnum = data.enumerateInstances();
        while (instEnum.hasMoreElements()) {
            Instance inst = (Instance) instEnum.nextElement();
            classCounts[(int)inst.classValue()]++;
        }

        for (double classCount : classCounts) {
            if (classCount > 0) {
                double p = classCount / numInstances;
                impurity -= p * p;
            }
        }

        return impurity;
    }
}
