package ml_6002b_coursework;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.Arrays;
import java.util.Enumeration;

import static ml_6002b_coursework.AttributeMeasures.measureChiSquared;
import static ml_6002b_coursework.AttributeMeasures.measureChiSquaredYates;

public class ChiSquaredAttributeSplitMeasure implements AttributeSplitMeasure {
    public static void main(String[] args) throws Exception{
        //load in dataset
        Instances data = WekaTools.loadClassificationData("src/main/java/ml_6002b_coursework/test_data/meningitis.arff");
        double attQuality;
        //no yates
        Enumeration attEnum = data.enumerateAttributes();
        while (attEnum.hasMoreElements()) {
            Attribute att = (Attribute) attEnum.nextElement();
            attQuality = new ChiSquaredAttributeSplitMeasure().computeAttributeQuality(data, att);
            System.out.println("Measure Chi-squared statistic for Attribute " + att.name() + " splitting diagnosis: " + attQuality);
        }
        //yates
        attEnum = data.enumerateAttributes();
        while (attEnum.hasMoreElements()) {
            Attribute att = (Attribute) attEnum.nextElement();
            attQuality = new ChiSquaredAttributeSplitMeasure(true).computeAttributeQuality(data, att);
            System.out.println("Measure Chi-squared yates statistic for Attribute " + att.name() + " splitting diagnosis: " + attQuality);
        }
    }


    boolean yates = false;

    public ChiSquaredAttributeSplitMeasure() { }

    public ChiSquaredAttributeSplitMeasure(boolean yates) {
        this.yates = yates;
    }

    public boolean isYates() {
        return yates;
    }

    @Override
    public double computeAttributeQuality(Instances data, Attribute att) {
        if (data.numInstances() == 0)
            return 0;
        Instances[] splitData = splitData(data, att);
        //use different number of values based on if attribute is nominal
        int numValues;
        if(att.isNominal()) {
            numValues = att.numValues();
        } else {
            numValues = splitData.length;
        }
        //create contingency table for attribute
        double[][] table = new double[numValues][data.numClasses()];
        for (int i = 0; i < att.numValues(); i++) {
            for (Instance instance : splitData[i]) {
                int value = (int) instance.classValue();
                table[i][value]++;
            }
        }
        if(yates) { return  measureChiSquaredYates(table); }
        else { return measureChiSquared(table); }

    }
}
