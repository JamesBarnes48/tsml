package ml_6002b_coursework;

import weka.classifiers.Classifier;
import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;

import java.util.Arrays;
import java.util.Enumeration;

import static java.lang.Double.max;

/**
 * Empty class for PArt 2.1 of the coursework
 *
 */
public class AttributeMeasures {
    public static void main(String[] args) throws Exception {
        //load in dataset
        Instances data = WekaTools.loadClassificationData("src/main/java/ml_6002b_coursework/test_data/meningitis.arff");
        Attribute att = data.attribute("headache");
        double[][] headacheTable = new double[][]{{3,2}, {3,4}};

        System.out.println("Measure IG for " + att.name() + ": Splitting diagnosis: "  + measureInformationGain(headacheTable));
        System.out.println("Measure Gini for " + att.name() + ": Splitting diagnosis: " + measureGini(headacheTable));
        System.out.println("Measure Chi-squared for " + att.name() + ": Splitting diagnosis: " + measureChiSquared(headacheTable));
        System.out.println("Measure Chi-squared with Yates correction for " + att.name() + ": Splitting diagnosis: " + measureChiSquaredYates(headacheTable));
    }

    //returns information gain for contingency table
    public static double measureInformationGain(double[][] split) throws Exception{
        if (split.length == 0 || split[0].length == 0)
            return 0;

        //find entropy of parent node
        double[] parent = new double[split[0].length];
        for (double[] attributeValue : split) {
            for (int j = 0; j < split[0].length; j++) {
                parent[j] += attributeValue[j];
            }
        }
        double infoGain = entropy(parent);

        //find entropy of child nodes
        double attSum = 0;
        double total = 0;
        for (double[] attributeValue : split) {
            double attributeValueCount = Arrays.stream(attributeValue).sum();
            if (attributeValueCount > 0){
                total += attributeValueCount * entropy(attributeValue);
                attSum += attributeValueCount;
            }
        }
        infoGain -= total / attSum;
        return infoGain;

    }

    private static double entropy(double[] split) throws Exception {
        // adapted from computeEntropy in ID3
        double entropy = 0;
        double count = 0;
        for (double classCount : split) {
            if (classCount > 0) {
                entropy -= classCount * Utils.log2(classCount);
                count += classCount;
            }
        }
        entropy /= count;
        return entropy + Utils.log2(count);
    }

    //returns gini measure for contingency table
    public static double measureGini(double[][] split) throws Exception{
        if (split.length == 0 || split[0].length == 0)
            return 0;

        //find gini of parent node
        double[] parent = new double[split[0].length];
        for (double[] attributeValue : split) {
            for (int j = 0; j < split[0].length; j++) {
                parent[j] += attributeValue[j];
            }
        }
        //add parent gini to total
        double gini = impurity(parent);

        //find gini for child nodes
        double count = 0;
        double sum = 0;
        for (double[] attributeValue : split) {
            double attributeValueCount = Arrays.stream(attributeValue).sum();
            if (attributeValueCount > 0){
                sum += attributeValueCount * impurity(attributeValue);
                count += attributeValueCount;
            }
        }
        gini -= sum / count;
        return gini;
    }

    private static double impurity(double[] split) throws Exception {
        double impurity = 0;
        double count = 0;
        for (double classCount : split) {
            if (classCount > 0) {
                impurity += classCount * classCount;
                count += classCount;
            }
        }

        impurity = 1 - impurity/(count * count);
        return impurity;
    }

    public static double calcChiSquared(double[][] split, boolean yates){
        double chi = 0;
        double count = 0;

        int attributeValues = split.length;
        int classes = split[0].length;
        if (attributeValues <= 1 && classes <= 1 || split[0].length == 0)
            return 0;

        double[] classTotals = new double [classes];
        double[] attTotals = new double [attributeValues];

        //calculate totals for row and column for calculating expected
        for (int row = 0; row < attributeValues; row++) {
            for (int col = 0; col < classes; col++) {
                double classCount = split[row][col];
                attTotals[row] += classCount;
                classTotals[col] += classCount;
                count += classCount;
            }
        }

        //traverse table and for each non-empty field
        for (int row = 0; row < attributeValues; row++) {
            if (attTotals[row] > 0) {
                for (int col = 0; col < classes; col++) {
                    if (classTotals[col] > 0) {
                        //calculate expected value for field, find difference and add to total chi
                        double expected = attTotals[row] * (classTotals[col] / count);
                        double diff = Math.abs(split[row][col] - expected);
                        //if using yates apply yates correction
                        if(yates) {
                            diff = max(diff - 0.5, 0);
                        }
                        chi += diff * diff / expected;
                    }
                }
            }
        }
        return chi;
    }

    public static double measureChiSquared(double[][] array) {
        return calcChiSquared(array, false);
    }

    public static double measureChiSquaredYates(double[][] array){
        return calcChiSquared(array, true);
    }

    public static double[][] contingencyTable(Instances data, Attribute att) {
        //System.out.println(att.numValues());
        //System.out.println("classes " + data.numClasses());
        double[][] table = new double[att.numValues()+1][data.numClasses()+1];
        //iterate through instances and add to class counts based on attribute att
        for(Instance ins:data) {
            table[(int) ins.value(att)][(int)ins.classValue()]++;
        }
        for(int i=0; i<data.numClasses(); i++) {
            for(int j=0; j<att.numValues(); j++) {
                table[att.numValues()][i] = table[att.numValues()][i] + table[j][i];
            }
        }
        //total for each attribute value
        for(int i=0; i<att.numValues(); i++) {
            for(int j=0; j<data.numClasses(); j++) {
                table[i][data.numClasses()] = table[i][data.numClasses()] + table[i][j];
            }
        }
        //table[0][2] = table[0][0] + table[0][1];
        //table[1][2] = table[1][0] + table[1][1];
        //System.out.println("contingency table for attribute " + att.name());
        //System.out.println("Rows - attribute values || Columns - class counts");
        //System.out.println(Arrays.deepToString(table).replace("], ", "]\n"));
        return table;
    }

    private static double[][] calcExpectedTable(int[][] contingencyTable, double[] globalProbs) {
        double[][] expectedTable = new double[contingencyTable.length][contingencyTable[0].length];
        //calculate expected number of cases
        for(int i=0; i<contingencyTable.length; i++) {
            for(int j=0; j<contingencyTable[0].length; j++) {
                expectedTable[i][j] = contingencyTable[i][contingencyTable[i].length] * globalProbs[j];
            }
        }
        return expectedTable;
    }
}
