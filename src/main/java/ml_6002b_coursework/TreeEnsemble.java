package ml_6002b_coursework;

import weka.classifiers.AbstractClassifier;
import weka.classifiers.meta.Bagging;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.RandomSubset;
import weka.filters.unsupervised.attribute.Remove;

import java.util.*;
import java.util.stream.IntStream;

import static utilities.Utilities.argMax;

public class TreeEnsemble
        extends AbstractClassifier
        implements OptionHandler, Randomizable, WeightedInstancesHandler,
        TechnicalInformationHandler {
    /**
     * The number of trees
     */
    protected int m_numTrees = 50;
    /**
     * Number of features
     */
    protected int m_numFeatures = 0;
    /**
     * proportion of attributes to sample in classifiers
     * Default 0.5
     */
    protected double m_sampleSize = 0.5;
    /**
     * The random seed.
     */
    protected int m_randomSeed = 2;

    private final LinkedHashMap<ID3Coursework, RandomSubset> usedAttributes = new LinkedHashMap<>();

    /**
     * The maximum depth of the trees (0 = unlimited)
     */
    protected int m_MaxDepth = 0;

    protected boolean averageDistribution = false;

    private ID3Coursework classifier = new ID3Coursework();

    /**
     * Returns a string describing classifier
     *
     * @return a description suitable for
     * displaying in the explorer/experimenter gui
     */
    public String globalInfo() {

        return
                "Class for constructing a forest of random trees.\n\n"
                        + "For more information see: \n\n"
                        + getTechnicalInformation().toString();
    }

    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation result;

        result = new TechnicalInformation(TechnicalInformation.Type.ARTICLE);
        result.setValue(TechnicalInformation.Field.AUTHOR, "James Barnes");
        result.setValue(TechnicalInformation.Field.YEAR, "2021");
        result.setValue(TechnicalInformation.Field.TITLE, "TreeEnsemble");

        return result;
    }

    /*
    ACCESSOR AND MUTATOR FUNCTIONS
     */
    public int getNumTrees() { return m_numTrees; }

    public void setNumTrees(int newNumTrees) { m_numTrees = newNumTrees; }

    public void setSeed(int seed) { m_randomSeed = seed; }

    public int getSeed() { return m_randomSeed; }

    public double getSampleSize() { return m_sampleSize; }

    public void setSampleSize(double sampleSize) { m_sampleSize = sampleSize; }

    public int getNumFeatures() { return m_numFeatures; }

    public void setNumFeatures(int newNumFeatures) { m_numFeatures = newNumFeatures; }

    public boolean getVoting() {
        return averageDistribution;
    }

    public void setVoting(boolean averageDistribution) {
        this.averageDistribution = averageDistribution;
    }


    public int getMaxDepth() { return m_MaxDepth; }

    public void setMaxDepth(int value) { m_MaxDepth = value; }

    /*
    GET AND SET OPTIONS
     */
    public String[] getOptions() {
        Vector result;
        String[] options;
        int i;

        result = new Vector();

        result.add("-I");
        result.add("" + getNumTrees());

        result.add("-K");
        result.add("" + getNumFeatures());

        result.add("-S");
        result.add("" + getSeed());

        result.add("-N");
        result.add("" + getSampleSize());

        if (getMaxDepth() > 0) {
            result.add("-depth");
            result.add("" + getMaxDepth());
        }


        options = super.getOptions();
        for (i = 0; i < options.length; i++)
            result.add(options[i]);

        return (String[]) result.toArray(new String[result.size()]);
    }

    public void setOptions(String[] options) throws Exception {
        String tmpStr;

        tmpStr = Utils.getOption('I', options);
        if (tmpStr.length() != 0) {
            m_numTrees = Integer.parseInt(tmpStr);
        } else {
            m_numTrees = 50;
        }

        tmpStr = Utils.getOption('K', options);
        if (tmpStr.length() != 0) {
            m_numFeatures = Integer.parseInt(tmpStr);
        } else {
            m_numFeatures = 0;
        }

        tmpStr = Utils.getOption('S', options);
        if (tmpStr.length() != 0) {
            setSeed(Integer.parseInt(tmpStr));
        } else {
            setSeed(2);
        }

        tmpStr = Utils.getOption("depth", options);
        if (tmpStr.length() != 0) {
            setMaxDepth(Integer.parseInt(tmpStr));
        } else {
            setMaxDepth(0);
        }

        //set sample size between 0 and 1
        tmpStr = Utils.getOption('N', options);
        try {
            if(tmpStr.length() != 0 && Double.parseDouble(tmpStr) > 0 && Double.parseDouble(tmpStr) < 1) {
                setSampleSize(Double.parseDouble(tmpStr));
            } else {
                setSampleSize(0.5);
            }
        } catch(Exception e) {
            System.out.println(e);
        }

        super.setOptions(options);

        Utils.checkForRemainingOptions(options);
    }

    private Remove getFilter(int[] att){
        Remove attGett = new Remove();
        attGett.setInvertSelection(true);
        attGett.setAttributeIndicesArray(att);
        return attGett;
    }

    @Override
    public void buildClassifier(Instances data) throws Exception {
        //initialise Random and set seed to reproduce results
        Random rand = new Random();
        rand.setSeed(m_randomSeed);
        //calculate number of attributes in sample
        /*int numAttributes = (int) ((data.numAttributes()) * m_sampleSize);
        //create sample datasets
        for(int i=0; i < m_numTrees; i++) {
            //System.out.println("NEW TREE");
            //randomly find indices of attributes to remove to create random sample
            int sampleIndices[] = new int[numAttributes];
            for(int j=0; j<(numAttributes); j++) {
                sampleIndices[j] = rand.nextInt(data.numAttributes()-1);
            }
            //remove attributes at the random indices from a dataset using filter
            Instances subset = data;
            Remove removeFilter = new Remove();
            removeFilter.setAttributeIndicesArray(sampleIndices);
            removeFilter.setInputFormat(data);
            subset = Filter.useFilter(data, removeFilter);
            subsets[i] = subset;

            //System.out.println("tree " + i);
            Enumeration attEnum = subset.enumerateAttributes();
            while (attEnum.hasMoreElements()) {
                Attribute att = (Attribute) attEnum.nextElement();
                //System.out.println(att.name());
            }
            //create a tree with the current filtered dataset
            classifiers[i] = new ID3Coursework();
            //create random options for tree
            String[] options = new String[2];
            options[0] = "-S";
            int randomChoice = rand.nextInt(4);
            switch(randomChoice) {
                case 0:
                    options[1] = "i";
                    break;
                case 1:
                    options[1] = "g";
                    break;
                case 2:
                    options[1] = "c";
                    break;
                case 3:
                    options[1] = "y";
                    break;
            }
            classifiers[i].setOptions(options);
            //---set options here---//
            classifiers[i].buildClassifier(subsets[i]);
        }
         */
        for (int i = 0; i < m_numTrees; i++) {
            RandomSubset attIndices = new RandomSubset();
            attIndices.setSeed(m_randomSeed + i);
            attIndices.setNumAttributes(m_sampleSize);
            attIndices.setInputFormat(data);
            Instances ins = attIndices.process(data);
            ID3Coursework c;
            c = classifier;
            //create random options for tree
            String[] options = new String[2];
            options[0] = "-S";
            int randomChoice = rand.nextInt(4);
            switch(randomChoice) {
                case 0:
                    options[1] = "i";
                    break;
                case 1:
                    options[1] = "g";
                    break;
                case 2:
                    options[1] = "c";
                    break;
                case 3:
                    options[1] = "y";
                    break;
            }
            c.setOptions(options);
            c.buildClassifier(ins);
            usedAttributes.put(c, attIndices);
        }
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        //use chosen voting scheme to classify instance
        return argMax(distributionForInstance(instance), new Random());
    }


    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        double[] distribution = new double[instance.numClasses()];
        usedAttributes.forEach((c, attributes) -> {
            try {
                attributes.setInputFormat(instance.dataset());
                attributes.input(instance);
                Instance ins = attributes.output();
                if (averageDistribution){
                    double[] distr = c.distributionForInstance(ins);
                    IntStream.range(0, distr.length).forEach(i -> distribution[i] += distr[i]);
                }
                else{
                    distribution[(int) c.classifyInstance(ins)]++;
                }
            } catch (Exception e) {
                e.printStackTrace();
            }
        });
        IntStream.range(0, distribution.length).forEach(i -> distribution[i] /= m_numTrees);
        return distribution;
    }


    public static void main(String[] args) throws Exception{
        //load in dataset
        Instances data = WekaTools.loadClassificationData("src/main/java/ml_6002b_coursework/test_data/optdigits.arff");
        Instances[] split = WekaTools.splitData(data, 0.7);
        TreeEnsemble c = new TreeEnsemble();
        c.buildClassifier(split[0]);
        System.out.println("Ensemble successfully built on data");
        //set options
        c.setVoting(true);
        //calculate accuracy
        double acc = WekaTools.accuracy(c, split[1]);
        System.out.println("Test accuracy: " + acc);
        //print predictions for first 5 instances
        Enumeration insenum = split[1].enumerateInstances();
        for(int i=0; i<5; i++) {
            Instance ins = (Instance) insenum.nextElement();
            double[] dis = c.distributionForInstance(ins);
            System.out.println("Probability distribution for instance:");
            System.out.println(c.classifyInstance(ins));
            //System.out.println(Arrays.toString(dis));
        }
    }


}
