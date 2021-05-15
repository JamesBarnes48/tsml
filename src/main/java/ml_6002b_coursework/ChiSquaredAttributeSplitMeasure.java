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
        //set target attribute
        data.setClassIndex(data.numAttributes()-1);
        //compute attribute quality for each attribute on base dataset and print to usesr
        Enumeration attEnum = data.enumerateAttributes();
        double attQuality;
        while (attEnum.hasMoreElements()) {
            Attribute att = (Attribute) attEnum.nextElement();
            attQuality = new ChiSquaredAttributeSplitMeasure().computeAttributeQuality(data, att);
            System.out.println("Measure Chi-squared statistic for Attribute " + att.name() + " splitting diagnosis: " + attQuality);
        }
        /*
        try {
            ID3Coursework c = new ID3Coursework();
            String[] options = {"-S", "c"};
            c.setOptions(options);
            c.buildClassifier(data);
        }
        catch(Exception e) {
            System.out.println("Exception: " + e);
        }*/
    }


    boolean yates = false;

    public ChiSquaredAttributeSplitMeasure() {
    }

    public ChiSquaredAttributeSplitMeasure(boolean yates) {
        this.yates = yates;
    }

    public boolean isYates() {
        return yates;
    }

    /**
     * Computes Chi Squared statistic for an attribute.
     *
     * @param data the data for which chi is to be computed
     * @param att  the attribute
     * @return the chi for the given attribute and data
     */
    @Override
    public double computeAttributeQuality(Instances data, Attribute att) {
        if (data.numInstances() == 0)
            return 0;
        Instances[] splitData = splitData(data, att);
        int numValues = att.isNominal() ? att.numValues() : splitData.length;
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

    @Override
    public String toString() {
        if (yates) {
            return "-Y: Attribute is Chi Squared statistic with Yates correction.";
        } else {
            return "-C: Attribute is Chi Squared statistic.";
        }

    }
}
