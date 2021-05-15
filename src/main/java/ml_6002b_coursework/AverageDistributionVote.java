package ml_6002b_coursework;

import weka.core.Instance;

import java.util.*;

public class AverageDistributionVote implements VotingSystem {
    @Override
    public double countVotes(TreeEnsemble c, Instance instance) throws Exception {
        //get array of class names to refer to
        double[] classNames = new double[instance.numClasses()];
        for (int i=0;i<instance.classAttribute().numValues();i++) {
            classNames[i] = Double.parseDouble(instance.classAttribute().value(i));
        }
        //initialise sum of distributions map to hold total of distribution values before finding mean average
        double[][] distributions = new double[c.m_numTrees][instance.numClasses()];
        //populate distributions
        for(int i=0; i<c.m_numTrees; i++) {
            double[] currentDist = c.classifiers[i].distributionForInstance(instance);
            distributions[i] = currentDist;
        }
        //find sum of all distributions for each class
        double[] distributionSum = new double[instance.numClasses()];
        for(int i=0; i<instance.numClasses(); i++) {
            double count = 0;
            for(int j=0; j<c.m_numTrees; j++) {
                count = count + distributions[j][i];
            }
            distributionSum[i] = count;
        }
        //divide sum of distribution by total number of classifiers and find the highest value
        double largest = 0;
        double prediction = 0;
        for(int i=0; i<distributionSum.length; i++) {
            distributionSum[i] = distributionSum[i] / c.m_numTrees;
            if(distributionSum[i] > largest) {
                largest = distributionSum[i];
                prediction = classNames[i];
            }
        }
        return prediction;
        //poll classifiers for distribution
        /*
        //populate array of class names
        String[] classNames = new String[instance.numClasses()];
        Enumeration classes = instance.classAttribute().enumerateValues();
        int counter = 0;
        while(classes.hasMoreElements()) {
            classNames[counter] = (String) classes.nextElement();
            counter++;
        }
        System.out.println(Arrays.toString(classNames));
        double[][] dist = new double[c.m_numTrees][instance.numClasses()];
        System.out.println(c.m_numTrees);
        //get distribution of instance
        for(int i=0; i<c.classifiers.length; i++) {
            dist[i] = c.classifiers[i].distributionForInstance(instance);
        }
        System.out.println(Arrays.deepToString(dist)); */
        //find average across classes and return largest class
    }
}
