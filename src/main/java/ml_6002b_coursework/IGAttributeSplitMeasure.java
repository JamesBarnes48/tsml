package ml_6002b_coursework;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.util.Enumeration;

public class IGAttributeSplitMeasure implements AttributeSplitMeasure{

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
            attQuality = new IGAttributeSplitMeasure().computeAttributeQuality(data, att);
            System.out.println("Measure IG for Attribute " + att.name() + " splitting diagnosis: " + attQuality);
        }
    }

    @Override
    public double computeAttributeQuality(Instances data, Attribute att) {
        double infoGain = computeEntropy(data);
        Instances[] splitData = splitData(data, att);
        for (int j = 0; j < att.numValues(); j++) {
            if (splitData[j].numInstances() > 0) {
                infoGain -= ((double) splitData[j].numInstances() /
                        (double) data.numInstances()) *
                        computeEntropy(splitData[j]);
            }
        }
        return infoGain;
    }

    private static double computeEntropy(Instances data) {
        double [] classCounts = new double[data.numClasses()];
        Enumeration instEnum = data.enumerateInstances();
        while (instEnum.hasMoreElements()) {
            Instance inst = (Instance) instEnum.nextElement();
            classCounts[(int) inst.classValue()]++;
        }
        double entropy = 0;
        for (int j = 0; j < data.numClasses(); j++) {
            if (classCounts[j] > 0) {
                entropy -= classCounts[j] * Utils.log2(classCounts[j]);
            }
        }
        entropy /= (double) data.numInstances();
        return entropy + Utils.log2(data.numInstances());
    }


}
