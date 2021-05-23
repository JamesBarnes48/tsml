/*
 *    This program is free software; you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation; either version 2 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program; if not, write to the Free Software
 *    Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.
 */

/*
 *    Id3.java
 *    Copyright (C) 1999 University of Waikato, Hamilton, New Zealand
 *
 */

package ml_6002b_coursework;

import com.sun.scenario.effect.impl.sw.sse.SSEBlend_SRC_OUTPeer;
import org.yaml.snakeyaml.events.Event;
import scala.Array;
import scala.Int;
import scala.None;
import weka.classifiers.AbstractClassifier;
import weka.classifiers.Sourcable;
import weka.core.*;
import weka.core.Capabilities.Capability;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;

import javax.rmi.CORBA.Util;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.Map;
import java.util.Random;

import static ml_6002b_coursework.WekaTools.loadClassificationData;
import static ml_6002b_coursework.WekaTools.splitData;
import static utilities.InstanceTools.resampleInstances;

/**

* Adaptation of the Id3 Weka classifier for use in machine learning coursework (6002B)

 <!-- globalinfo-start -->
 * Class for constructing an unpruned decision tree based on the ID3 algorithm. Can only deal with nominal attributes. No missing values allowed. Empty leaves may result in unclassified instances. For more information see: <br/>
 * <br/>
 * R. Quinlan (1986). Induction of decision trees. Machine Learning. 1(1):81-106.
 * <p/>
 <!-- globalinfo-end -->
 *
 <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;article{Quinlan1986,
 *    author = {R. Quinlan},
 *    journal = {Machine Learning},
 *    number = {1},
 *    pages = {81-106},
 *    title = {Induction of decision trees},
 *    volume = {1},
 *    year = {1986}
 * }
 * </pre>
 * <p/>
 <!-- technical-bibtex-end -->
 *
 <!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -D
 *  If set, classifier is run in debug mode and
 *  may output additional info to the console</pre>
 * 
 <!-- options-end -->
 *
 * @author Eibe Frank (eibe@cs.waikato.ac.nz)
 * @version $Revision: 6404 $ 
 */
public class ID3Coursework
  extends AbstractClassifier {

  /** for serialization */
  static final long serialVersionUID = -2693678647096322561L;
  
  /** The node's successors. */ 
  private ID3Coursework[] m_Successors;

  /** Attribute used for splitting. */
  private Attribute m_Attribute;

  /** Max tree depth. default none (-1) */
  private int m_maxDepth = -1;

  /**
   * random value used for splitting numeric attributes.
   */
  private double randomSplitValue;

  /** Class value if node is leaf. */
  private double m_ClassValue;

  /** Class distribution if node is leaf. */
  private double[] m_Distribution;

  /** Class attribute of dataset. */
  private Attribute m_ClassAttribute;
  private AttributeSplitMeasure attSplit = new IGAttributeSplitMeasure();

  /** Accessor and mutator for options */
  public int getMaxDepth() { return m_maxDepth; }
  public void setMaxDepth(int depth) {
    m_maxDepth = depth;
  }
  //public String

  /**
   * Returns a string describing the classifier.
   * @return a description suitable for the GUI.
   */
  public String globalInfo() {

    return  "Class for constructing an unpruned decision tree based on the ID3 "
      + "algorithm. Can only deal with nominal attributes. No missing values "
      + "allowed. Empty leaves may result in unclassified instances. For more "
      + "information see: \n\n"
      + getTechnicalInformation().toString();
  }

  /**
   * Returns an instance of a TechnicalInformation object, containing 
   * detailed information about the technical background of this class,
   * e.g., paper reference or book this class is based on.
   * 
   * @return the technical information about this class
   */
  public TechnicalInformation getTechnicalInformation() {
    TechnicalInformation 	result;
    
    result = new TechnicalInformation(Type.ARTICLE);
    result.setValue(Field.AUTHOR, "R. Quinlan");
    result.setValue(Field.YEAR, "1986");
    result.setValue(Field.TITLE, "Induction of decision trees");
    result.setValue(Field.JOURNAL, "Machine Learning");
    result.setValue(Field.VOLUME, "1");
    result.setValue(Field.NUMBER, "1");
    result.setValue(Field.PAGES, "81-106");
    
    return result;
  }

  //return attribute value based on if attribute is numeric or not
  private int branch(Instance instance){
    int temp;
    if(m_Attribute.isNumeric()) {
      //return binned value based on binary split
      if(instance.value(m_Attribute) < randomSplitValue) {
        temp = 0;
      } else {
        temp = 1;
      }
    }
    //return discrete value of attribute
    else {
      temp = (int)instance.value(m_Attribute);
    }
    return temp;
  }

  /**
   * Returns default capabilities of the classifier.
   *
   * @return      the capabilities of this classifier
   */
  public Capabilities getCapabilities() {
    Capabilities result = super.getCapabilities();
    result.disableAll();

    // attributes
    result.enable(Capability.NOMINAL_ATTRIBUTES);
    result.enable(Capability.NUMERIC_ATTRIBUTES);

    // class
    result.enable(Capability.NOMINAL_CLASS);
    result.enable(Capability.NUMERIC_CLASS);
    result.enable(Capability.MISSING_CLASS_VALUES);

    // instances
    result.setMinimumNumberInstances(0);
    
    return result;
  }

  /* SET OPTIONS
  Sets options for building the tree
  Allows the splitting criteria to be set
  '-S'   -   splitting criteria option flag

  'i'   -   information gain
  'g'   -   gini
  'c'   -   chi-squared
  'y'   -   chi-squared yates

   */

  public void setOptions(String[] options) throws Exception {
    String splitCriteria = Utils.getOption('S', options);
    if(splitCriteria.equals("i")) {
      attSplit = new IGAttributeSplitMeasure();
    }
    else if(splitCriteria.equals("g")) {
      attSplit = new GiniAttributeSplitMeasure();
    }
    else if(splitCriteria.equals("c")) {
      attSplit = new ChiSquaredAttributeSplitMeasure();
    }
    else if(splitCriteria.equals("y")) {
      attSplit = new ChiSquaredAttributeSplitMeasure(true);
    }
    //default case if none satisfied
    else {
      System.out.println("Split criteria option not found: " + splitCriteria + ". Set to information gain by default");
      attSplit = new IGAttributeSplitMeasure();
    }
  }

//retrieve options for classifier
  public String[] getOptions() {
    String[] options = new String[2];
    options[0] = "-S";
    if(attSplit instanceof IGAttributeSplitMeasure) {
      options[1] = "i";
    }
    else if(attSplit instanceof  GiniAttributeSplitMeasure) {
      options[1] = "g";
    }
    else if(attSplit instanceof  ChiSquaredAttributeSplitMeasure) {
      if(((ChiSquaredAttributeSplitMeasure) attSplit).isYates()) {
        options[1] = "y";
      } else {
        options[1] = "c";
      }
    }
    return options;
  }

  //replaces set options
  public void setSplitMeasure(char att) {
    switch(att) {
      case 'I': attSplit = new IGAttributeSplitMeasure(); break;
      case 'G': attSplit = new GiniAttributeSplitMeasure(); break;
      case 'C': attSplit = new ChiSquaredAttributeSplitMeasure(); break;
      case 'Y': attSplit = new ChiSquaredAttributeSplitMeasure(true); break;
      default:
        System.out.println("Invalid split measure - set to information gain by default");
        attSplit = new IGAttributeSplitMeasure();
    }
  }

  //return split measure as char for recursively setting split measure
  public char getSplitMeasureAsChar() {
    if(attSplit instanceof IGAttributeSplitMeasure) {
      return 'I';
    }
    else if(attSplit instanceof GiniAttributeSplitMeasure) {
      return 'G';
    }
    else if(attSplit instanceof  ChiSquaredAttributeSplitMeasure) {
      if(((ChiSquaredAttributeSplitMeasure) attSplit).isYates()) {
        return 'Y';
      } else {
        return 'C';
      }
    }
    else {
      System.out.println("Split measure not found");
      return 'f';
    }
  }

  /**
   * Builds Id3 decision tree classifier.
   *
   * @param data the training data
   * @exception Exception if classifier can't be built successfully
   */
  public void buildClassifier(Instances data) throws Exception {

    // can classifier handle the data?
    getCapabilities().testWithFail(data);

    // remove instances with missing class
    data = new Instances(data);
    data.deleteWithMissingClass();
    
    makeTree(data, m_maxDepth);
  }

  /**
   * Method for building an Id3 tree.
   *
   * @param data the training data
   * @exception Exception if decision tree can't be built successfully
   */
  private void makeTree(Instances data, int maxDepth) throws Exception {

    int numInstances = data.numInstances();
    int numClasses = data.numClasses();
    Attribute classAtt = data.classAttribute();
    // Check if no instances have reached this node.
    if (numInstances == 0) {
      m_Attribute = null;
      m_ClassValue = Utils.missingValue();
      m_Distribution = new double[numClasses];
      return;
    }
    else if (numInstances == 1 || classAtt.numValues() == 1) {
      m_Attribute = null;
      m_ClassValue = data.get(0).classValue();
      m_Distribution = new double[numClasses];
      m_Distribution[(int) m_ClassValue] = numInstances;
      m_ClassAttribute = classAtt;

      return;
    }

    // Compute attribute with maximum information gain.
    double[] infoGains = new double[data.numAttributes()];
    Enumeration attEnum = data.enumerateAttributes();
    while (attEnum.hasMoreElements()) {
      Attribute att = (Attribute) attEnum.nextElement();
      infoGains[att.index()] = attSplit.computeAttributeQuality(data, att);
    }
    m_Attribute = data.attribute(Utils.maxIndex(infoGains));

    // Create leaf if info gain is zero OR max depth reached
    // stop when there are no more attributes to split / there is only one class
    if (Utils.eq(infoGains[m_Attribute.index()], 0) || maxDepth == 0) {
      m_Attribute = null;
      m_Distribution = new double[numClasses];
      Enumeration insEnum = data.enumerateInstances();
      while (insEnum.hasMoreElements()) {
        Instance ins = (Instance) insEnum.nextElement();
        m_Distribution[(int) ins.classValue()]++;
      }
      Utils.normalize(m_Distribution);
      m_ClassValue = Utils.maxIndex(m_Distribution);
      m_ClassAttribute = classAtt;
    }
    //create successors if info gain is not zero
    else {
      //perform different splits based on attribute type
      Instances[] splitData = new Instances[0];
      if (m_Attribute.isNominal())
        splitData = attSplit.splitData(data, m_Attribute);
      else if (m_Attribute.isNumeric()){
        Map.Entry<Instances[], Double> splitMap = attSplit.splitDataOnNumeric(data, m_Attribute);
        splitData = splitMap.getKey();
        randomSplitValue = splitMap.getValue();
      }
      int numValues;
      if(m_Attribute.isNominal()) {
        numValues = m_Attribute.numValues();
      } else {
        numValues = splitData.length;
      }
      //create successor nodes and recursive call
      m_Successors = new ID3Coursework[numValues];
      for (int j = 0; j < numValues; j++) {
        m_Successors[j] = new ID3Coursework();
        m_Successors[j].setSplitMeasure(this.getSplitMeasureAsChar());
        m_Successors[j].makeTree(splitData[j], maxDepth-1);
      }
    }
  }

  /**
   * Classifies a given test instance using the decision tree.
   *
   * @param instance the instance to be classified
   * @return the classification
   * @throws NoSupportForMissingValuesException if instance has missing values
   */
  public double classifyInstance(Instance instance) 
    throws NoSupportForMissingValuesException {

    if (instance.hasMissingValue()) {
      throw new NoSupportForMissingValuesException("Please ensure there are no missing values");
    }
    //base case - null attribute = leaf node
    if (m_Attribute == null) {
      return m_ClassValue;
    }
    //recursively call classifyInstance on the correct successor node
    else {
      return m_Successors[branch(instance)].
              classifyInstance(instance);
    }
  }

  /**
   * Computes class distribution for instance using decision tree.
   *
   * @param instance the instance for which distribution is to be computed
   * @return the class distribution for the given instance
   * @throws NoSupportForMissingValuesException if instance has missing values
   */
  public double[] distributionForInstance(Instance instance) 
    throws NoSupportForMissingValuesException {

    if (instance.hasMissingValue()) {
      throw new NoSupportForMissingValuesException("Please ensure there are no missing values");
    }
    if (m_Attribute == null) {
      return m_Distribution;
    } else {
      int temp = branch(instance);
      return m_Successors[temp].
        distributionForInstance(instance);
    }
  }

  public String getAttName() {
    return attSplit.getClass().getSimpleName();
  }

  /**
   * Main method.
   *
   * @param args the options for the classifier
   */
  ////////////////MAIN METHOD
  public static void main(String[] args) throws Exception {

    Instances chinatownTrain = loadClassificationData("src/main/java/ml_6002b_coursework/test_data/Chinatown_TRAIN.arff");
    Instances chinatownTest = loadClassificationData("src/main/java/ml_6002b_coursework/test_data/Chinatown_TEST.arff");

    Instances optdigits = loadClassificationData("src/main/java/ml_6002b_coursework/test_data/optdigits.arff");
    Instances[] trainTest = splitData(optdigits, 0.7);
    Instances optdigitsTrain = trainTest[0];
    Instances optdigitsTest = trainTest[1];

    //Instances diagnosis = loadClassificationData("src/main/java/ml_6002b_coursework/test_data/Diagnosis_TRAIN.arff");

    try{
      ID3Coursework id3 = new ID3Coursework();

      //OPTDIGITS
      //use infogain
      //initialise options
      //String[] options = new String[2];
      //options[0] = "-S";
      //options[1] = "i";
      //id3.setOptions(options);
      id3.setSplitMeasure('I');
      id3.setMaxDepth(-1);
      id3.buildClassifier(optdigitsTrain);
      //id3.buildClassifier(chinatownTrain);
      System.out.println("Id3 using measure " + id3.getAttName() + " on JW Problem has test accuracy = "
              + WekaTools.accuracy(id3, optdigitsTest));

      //use gini
      //options = new String[2];
      //options[0] = "-S";
      //options[1] = "g";
      //id3.setOptions(options);
      id3.setSplitMeasure('G');
      id3.setMaxDepth(-1);
      //id3.buildClassifier(chinatownTrain);
      id3.buildClassifier(optdigitsTrain);
      System.out.println("Id3 using measure " + id3.getAttName() + " on JW Problem has test accuracy = "
              + WekaTools.accuracy(id3, optdigitsTest));

      //use chisquared
      //options = new String[2];
      //options[0] = "-S";
      //options[1] = "c";
      //id3.setOptions(options);
      id3.setSplitMeasure('C');
      id3.setMaxDepth(-1);
      id3.buildClassifier(optdigitsTrain);
      System.out.println("Id3 using measure " + id3.getAttName() + " on JW Problem has test accuracy = "
              + WekaTools.accuracy(id3, optdigitsTest));

      //use chisquared yates
      //options = new String[2];
      //options[0] = "-S";
      //options[1] = "y";
      //id3.setOptions(options);
      id3.setSplitMeasure('Y');
      id3.setMaxDepth(-1);
      id3.buildClassifier(optdigitsTrain);
      System.out.println("Id3 using measure " + id3.getAttName() + " with Yates on JW Problem has test accuracy = "
              + WekaTools.accuracy(id3, optdigitsTest));

      //CHINATOWN
      //use infogain
      //options = new String[2];
      //options[0] = "-S";
      //options[1] = "i";
      //id3.setOptions(options);
      id3.setSplitMeasure('I');
      id3.setMaxDepth(-1);
      id3.buildClassifier(chinatownTrain);
      System.out.println("Id3 using measure " + id3.getAttName() + " on Chinatown Problem has test accuracy = "
              + WekaTools.accuracy(id3, chinatownTest));

      //use gini
      //options = new String[2];
      //options[0] = "-S";
      //options[1] = "g";
      //id3.setOptions(options);
      id3.setSplitMeasure('G');
      id3.setMaxDepth(-1);
      id3.buildClassifier(chinatownTrain);
      System.out.println("Id3 using measure " + id3.getAttName() + " on Chinatown Problem has test accuracy = "
              + WekaTools.accuracy(id3, chinatownTest));

      //use chisquared
      //options = new String[2];
      //options[0] = "-S";
      //options[1] = "c";
      //id3.setOptions(options);
      id3.setSplitMeasure('C');
      id3.setMaxDepth(-1);
      id3.buildClassifier(chinatownTrain);
      System.out.println("Id3 using measure " + id3.getAttName() + " on Chinatown Problem has test accuracy = "
              + WekaTools.accuracy(id3, chinatownTest));

      //use chisquared yates
      //options = new String[2];
      //options[0] = "-S";
      //options[1] = "y";
      //id3.setOptions(options);
      id3.setSplitMeasure('Y');
      id3.setMaxDepth(-1);
      id3.buildClassifier(chinatownTrain);
      System.out.println("Id3 using measure " + id3.getAttName() + " with Yates on Chinatown Problem has test accuracy = "
              + WekaTools.accuracy(id3, chinatownTest));

    }
    catch (Exception e){
      e.printStackTrace();
    }
  }
}
