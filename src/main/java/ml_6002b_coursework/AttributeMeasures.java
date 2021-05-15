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
        data.setClassIndex(data.numAttributes()-1);
        Attribute att = data.attribute(0);
        int[][] table = contingencyTable(data, att);
        ////////
        /////////
        //CHANGE THIS TO USE THE METHODS IN THIS CLASS
        IGAttributeSplitMeasure ig = new IGAttributeSplitMeasure();
        double informationGain = ig.computeAttributeQuality(data, att);
        System.out.println("Measure IG for " + att.name() + ": Splitting diagnosis: "  + informationGain);
        GiniAttributeSplitMeasure gini = new GiniAttributeSplitMeasure();
        double ginival = gini.computeAttributeQuality(data, att);
        System.out.println("Measure Gini for " + att.name() + ": Splitting diagnosis: " + ginival);
        ChiSquaredAttributeSplitMeasure chi = new ChiSquaredAttributeSplitMeasure();
        double chiSquared = chi.computeAttributeQuality(data,att);
        System.out.println("Measure Chi-squared for " + att.name() + ": Splitting diagnosis: " + chiSquared);
        ChiSquaredAttributeSplitMeasure yate = new ChiSquaredAttributeSplitMeasure(true);
        double chiSquaredYates = yate.computeAttributeQuality(data,att);
        System.out.println("Measure Chi-squared with Yates correction for " + att.name() + ": Splitting diagnosis: " + chiSquaredYates);
        //train test split dataset
        /*
        Instances[] trainTestSplit = WekaTools.splitData(data, 0.6);
        Instances train = trainTestSplit[0];
        Instances test = trainTestSplit[1];
        //set target attribute
        train.setClassIndex(train.numAttributes()-1);
        test.setClassIndex(train.numAttributes()-1);

        try {
            Classifier c = new ID3Coursework();
            c.buildClassifier(train);
            int[] pred = WekaTools.classifyInstances(c, test);
            //System.out.println(Arrays.toString(pred));
            //System.out.println(Arrays.deepToString(WekaTools.confusionMatrix(pred, WekaTools.getClassValues(test))).replace("], ", "]\n"));
        }
        catch(Exception e) {
            System.out.println("Exception: " + e);
        }*/
    }

    public static double measureInformationGain(double[][] array){
        double returnValue = 0, rowSum, total = 0;
        int numRows = array.length;
        int numCols = array[0].length;

        for (double[] doubles : array) {
            rowSum = 0;
            for (int j = 0; j < numCols; j++) {
                returnValue = returnValue + (doubles[j] * Math.log(doubles[j]));
                rowSum += doubles[j];
            }
            returnValue = returnValue - (rowSum * Math.log(rowSum));
            total += rowSum;
        }
        try{
            return 1-(-returnValue / (total * Math.log(2)));
        }
        catch (Exception e){
            e.printStackTrace();
            return 0.0;
        }

    }


    public static double measureGini(double[][] array){
        try{
            double returnValue = 0, rowSum = 0, total = 0, weighted = 0;
            int numRows = array.length;
            int numCols = array[0].length;
            double[] values = new double[numCols];

            for (double[] doubles : array) {
                for (int j = 0; j < numCols; j++) {
                    total += doubles[j];
                }
            }


            for (double[] doubles : array) {
                returnValue = 0;
                rowSum = 0;
                for (int j = 0; j < numCols; j++) {
                    rowSum += doubles[j];
                    values[j] = doubles[j];
                }

                for (double x : values){
                    //System.out.println(x + " / " + rowSum);
                    returnValue += Math.pow((x/rowSum),2);
                    //System.out.println("Return value " + returnValue);
                }


                returnValue = (1 - returnValue);
                //System.out.println("Calculated impurity for node  " + returnValue);
                //System.out.println("Calculation  " + returnValue + " * " + "( " + rowSum + " / " + total);
                weighted+= (returnValue*(rowSum/total));
                //System.out.println("Current weighted = " + weighted);
            }

            if (total == 0) {
                return 0;
            }

            return weighted;

        }
        catch (Exception e){
            e.printStackTrace();
            return 0.0;
        }
    }

    public static double calcChiSquared(double[][] split, boolean yates){
        double chi = 0;
        double n = 0;

        int attributeValues = split.length;
        int classes = split[0].length;
        if (attributeValues <= 1 && classes <= 1 || split[0].length == 0)
            return 0;

        double[] valueTotals = new double [attributeValues];
        double[] classTotals = new double [classes];

        for (int row = 0; row < attributeValues; row++) {
            for (int col = 0; col < classes; col++) {
                double classCount = split[row][col];
                valueTotals[row] += classCount;
                classTotals[col] += classCount;
                n += classCount;
            }
        }

        for (int row = 0; row < attributeValues; row++) {
            if (valueTotals[row] > 0) {
                for (int col = 0; col < classes; col++) {
                    if (classTotals[col] > 0) {
                        double expected = valueTotals[row] * (classTotals[col] / n);
                        double diff = Math.abs(split[row][col] - expected);
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

    public static int[][] contingencyTable(Instances data, Attribute att) {
        //System.out.println(att.numValues());
        //System.out.println("classes " + data.numClasses());
        int[][] table = new int[att.numValues()+1][data.numClasses()+1];
        //iterate through instances and add to class counts based on attribute att
        for(Instance ins:data) {
            table[(int) ins.value(att)][(int)ins.classValue()]++;
        }
        //find totals
        //total for each class
        //table[att.numValues()][0] = table[0][0] + table[1][0];
        //table[att.numValues()][1] = table[0][1] + table[1][1];
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
        /*
        expectedTable[0][0] = contingencyTable[0][2] * globalProbNegative;
        expectedTable[1][0] = contingencyTable[1][2] * globalProbNegative;
        expectedTable[0][1] = contingencyTable[0][2] * globalProbPositive;
        expectedTable[1][1] = contingencyTable[1][2] * globalProbPositive; */
        return expectedTable;
    }
}
