package ml_6002b_coursework;

import org.yaml.snakeyaml.events.Event;
import tsml.classifiers.distance_based.utils.collections.tree.Tree;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.ConfusionMatrix;
import weka.classifiers.meta.RotationForest;
import weka.classifiers.trees.Id3;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.SimpleCart;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;

import java.util.*;

public class Experiment {

    public static void main(String[] args) throws Exception {
        String path = "src/main/java/ml_6002b_coursework/test_data/";
        ///////////
        //SET DATASET HERE
        //String fileName = "yeast";
        //String fileName = "ozone";
        //String fileName = "led-display";
        //String fileName = "contraceptive-method";
        String fileName = "EOGHorizontalSignal_TRAIN";
        Instances train = WekaTools.loadClassificationData(path + fileName + ".arff");
        fileName = "EOGHorizontalSignal_TEST";
        Instances test = WekaTools.loadClassificationData(path + fileName + ".arff");

        System.out.println("on dataset: " + fileName);

        //Instances data = WekaTools.loadClassificationData(path + fileName + ".arff");
        //Instances[] split = WekaTools.splitData(data, 0.7);

        //Instances train = split[0];
        //Instances test = split[1];

        Discretize filter = new Discretize();
        filter.setInputFormat(train);
        train = Filter.useFilter(train, filter);
        test = Filter.useFilter(test, filter);

        /////////////
        //SET CLASSIFIER HERE
        ID3Coursework c = new ID3Coursework();
        ID3Coursework e = new ID3Coursework();
        //TreeEnsemble e = new TreeEnsemble();
        //Id3 c = new Id3();
        //J48 c = new J48();
        //RandomForest e = new RandomForest();
        //RotationForest c = new RotationForest();
        //NaiveBayes e = new NaiveBayes();
        //SimpleCart c = new SimpleCart();

        /////////////
        //SET OPTIONS HERE
        char splitmeasure = 'C';
        c.setSplitMeasure(splitmeasure);
        System.out.println("splitmeasure: " + splitmeasure);
        splitmeasure = 'Y';
        e.setSplitMeasure(splitmeasure);
        System.out.println("splitmeasure: " + splitmeasure);
        //c.setMaxDepth(300);
        c.buildClassifier(train);

        //create confusion matrix
        double[] actual = test.attributeToDoubleArray(test.classIndex());
        double[] pred = new double[test.numInstances()];

        ///////// C
        int count = 0;
        Enumeration insenum = test.enumerateInstances();
        while(insenum.hasMoreElements()) {
            Instance cur = (Instance) insenum.nextElement();
            pred[count] = c.classifyInstance(cur);
            count++;
        }
        double[][] conf = WekaTools.confusionMatrix(pred, actual, train.numClasses());
        System.out.println("classifier: " + c.getClass().getSimpleName());
        System.out.println(Arrays.deepToString(conf).replace("], ", "]\n"));
        System.out.println("ACCURACY: " + WekaTools.accuracy(c, test));
        System.out.println("BALANCED ACCURACY: " + WekaTools.balancedAccuracy(conf));

        ////////// E

        //splitmeasure = 'I';
        //System.out.println("splitmeasure: " + splitmeasure);
        //e.setMaxDepth(300);
        e.buildClassifier(train);
        count = 0;
        insenum = test.enumerateInstances();
        while(insenum.hasMoreElements()) {
            Instance cur = (Instance) insenum.nextElement();
            pred[count] = e.classifyInstance(cur);
            count++;
        }
        conf = WekaTools.confusionMatrix(pred, actual, train.numClasses());
        System.out.println("other classifier: " + e.getClass().getSimpleName());
        System.out.println(Arrays.deepToString(conf).replace("], ", "]\n"));
        System.out.println("ACCURACY: " + WekaTools.accuracy(e, test));
        System.out.println("BALANCED ACCURACY: " + WekaTools.balancedAccuracy(conf));

        //String parameter = "info gain";
        //String outFile = "ID3CW";
        //WekaTools.generateTestResults(c, train, test, parameter, "src/main/java/ml_6002b_coursework/test_result/", outFile);
        //String testout = "ID3TestResults.csv";
    }

}
