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
  extends AbstractClassifier 
  implements TechnicalInformationHandler, Sourcable {

  /** for serialization */
  static final long serialVersionUID = -2693678647096322561L;
  
  /** The node's successors. */ 
  private ID3Coursework[] m_Successors;

  /** Attribute used for splitting. */
  private Attribute m_Attribute;

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

  SET TO INFORMATION GAIN BY DEFAULT
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
    
    makeTree(data);
  }

  /**
   * Method for building an Id3 tree.
   *
   * @param data the training data
   * @exception Exception if decision tree can't be built successfully
   */
  private void makeTree(Instances data) throws Exception {

    int numInstances = data.numInstances();
    int numClasses = data.numClasses();
    Attribute classAtt = data.classAttribute();
    // Check if no instances have reached this node.
    if (numInstances == 0) {
      Random rand= new Random();
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

    // Create leaf if info gain is zero.
    // stop when there are no more attributes to split / there is only one class
    if (Utils.eq(infoGains[m_Attribute.index()], 0)) {
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
      m_Successors = new ID3Coursework[numValues];
      for (int j = 0; j < numValues; j++) {
        m_Successors[j] = new ID3Coursework();
        m_Successors[j].setOptions(this.getOptions());
        m_Successors[j].makeTree(splitData[j]);
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
      throw new NoSupportForMissingValuesException("Id3: no missing values, "
              + "please.");
    }
    if (m_Attribute == null) {
      return m_ClassValue;
    } else {
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

  /**
   * Prints the decision tree using the private toString method from below.
   *
   * @return a textual description of the classifier
   */
  public String toString() {

    if ((m_Distribution == null) && (m_Successors == null)) {
      return "Tree does not exist";
    }
    return "Id3\n\n" + toString(0);
  }


  /**
   * Outputs a tree at a certain level.
   *
   * @param level the level at which the tree is to be printed
   * @return the tree as string at the given level
   */
  private String toString(int level) {

    StringBuffer text = new StringBuffer();
    
    if (m_Attribute == null) {
      if (Utils.isMissingValue(m_ClassValue)) {
        text.append(": null");
      } else {
        text.append(": " + m_ClassAttribute.value((int) m_ClassValue));
      } 
    } else {
      for (int j = 0; j < m_Attribute.numValues(); j++) {
        text.append("\n");
        for (int i = 0; i < level; i++) {
          text.append("|  ");
        }
        text.append(m_Attribute.name() + " = " + m_Attribute.value(j));
        text.append(m_Successors[j].toString(level + 1));
      }
    }
    return text.toString();
  }

  /**
   * Adds this tree recursively to the buffer.
   * 
   * @param id          the unqiue id for the method
   * @param buffer      the buffer to add the source code to
   * @return            the last ID being used
   * @throws Exception  if something goes wrong
   */
  protected int toSource(int id, StringBuffer buffer) throws Exception {
    int                 result;
    int                 i;
    int                 newID;
    StringBuffer[]      subBuffers;
    
    buffer.append("\n");
    buffer.append("  protected static double node" + id + "(Object[] i) {\n");
    
    // leaf?
    if (m_Attribute == null) {
      result = id;
      if (Double.isNaN(m_ClassValue)) {
        buffer.append("    return Double.NaN;");
      } else {
        buffer.append("    return " + m_ClassValue + ";");
      }
      if (m_ClassAttribute != null) {
        buffer.append(" // " + m_ClassAttribute.value((int) m_ClassValue));
      }
      buffer.append("\n");
      buffer.append("  }\n");
    } else {
      buffer.append("    checkMissing(i, " + m_Attribute.index() + ");\n\n");
      buffer.append("    // " + m_Attribute.name() + "\n");
      
      // subtree calls
      subBuffers = new StringBuffer[m_Attribute.numValues()];
      newID = id;
      for (i = 0; i < m_Attribute.numValues(); i++) {
        newID++;

        buffer.append("    ");
        if (i > 0) {
          buffer.append("else ");
        }
        buffer.append("if (((String) i[" + m_Attribute.index() 
            + "]).equals(\"" + m_Attribute.value(i) + "\"))\n");
        buffer.append("      return node" + newID + "(i);\n");

        subBuffers[i] = new StringBuffer();
        newID = m_Successors[i].toSource(newID, subBuffers[i]);
      }
      buffer.append("    else\n");
      buffer.append("      throw new IllegalArgumentException(\"Value '\" + i["
          + m_Attribute.index() + "] + \"' is not allowed!\");\n");
      buffer.append("  }\n");

      // output subtree code
      for (i = 0; i < m_Attribute.numValues(); i++) {
        buffer.append(subBuffers[i].toString());
      }
      subBuffers = null;
      
      result = newID;
    }
    
    return result;
  }
  
  /**
   * Returns a string that describes the classifier as source. The
   * classifier will be contained in a class with the given name (there may
   * be auxiliary classes),
   * and will contain a method with the signature:
   * <pre><code>
   * public static double classify(Object[] i);
   * </code></pre>
   * where the array <code>i</code> contains elements that are either
   * Double, String, with missing values represented as null. The generated
   * code is public domain and comes with no warranty. <br/>
   * Note: works only if class attribute is the last attribute in the dataset.
   *
   * @param className the name that should be given to the source class.
   * @return the object source described by a string
   * @throws Exception if the source can't be computed
   */
  public String toSource(String className) throws Exception {
    StringBuffer        result;
    int                 id;
    
    result = new StringBuffer();

    result.append("class " + className + " {\n");
    result.append("  private static void checkMissing(Object[] i, int index) {\n");
    result.append("    if (i[index] == null)\n");
    result.append("      throw new IllegalArgumentException(\"Null values "
        + "are not allowed!\");\n");
    result.append("  }\n\n");
    result.append("  public static double classify(Object[] i) {\n");
    id = 0;
    result.append("    return node" + id + "(i);\n");
    result.append("  }\n");
    toSource(id, result);
    result.append("}\n");

    return result.toString();
  }

  public String getAtt() {
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
      Instances[] splits = splitData(optdigits, 0.5);

      ID3Coursework id3 = new ID3Coursework();

      //OPTDIGITS
      //use infogain
      //initialise options
      String[] options = new String[2];
      options[0] = "-S";
      options[1] = "i";
      id3.setOptions(options);
      id3.buildClassifier(optdigitsTrain);
      //id3.buildClassifier(chinatownTrain);
      System.out.println("Id3 using measure " + id3.getAtt() + " on JW Problem has test accuracy = "
              + WekaTools.accuracy(id3, optdigitsTest));

      //use gini
      options = new String[2];
      options[0] = "-S";
      options[1] = "g";
      id3.setOptions(options);
      //id3.buildClassifier(chinatownTrain);
      id3.buildClassifier(optdigitsTrain);
      System.out.println("Id3 using measure " + id3.getAtt() + " on JW Problem has test accuracy = "
              + WekaTools.accuracy(id3, optdigitsTest));

      //use chisquared
      options = new String[2];
      options[0] = "-S";
      options[1] = "c";
      id3.setOptions(options);
      id3.buildClassifier(optdigitsTrain);
      System.out.println("Id3 using measure " + id3.getAtt() + " on JW Problem has test accuracy = "
              + WekaTools.accuracy(id3, optdigitsTest));

      //use chisquared yates
      options = new String[2];
      options[0] = "-S";
      options[1] = "y";
      id3.setOptions(options);
      id3.buildClassifier(optdigitsTrain);
      System.out.println("Id3 using measure " + id3.getAtt() + " with Yates on JW Problem has test accuracy = "
              + WekaTools.accuracy(id3, optdigitsTest));

      //CHINATOWN
      //use infogain
      options = new String[2];
      options[0] = "-S";
      options[1] = "i";
      id3.setOptions(options);
      id3.buildClassifier(chinatownTrain);
      System.out.println("Id3 using measure " + id3.getAtt() + " on Chinatown Problem has test accuracy = "
              + WekaTools.accuracy(id3, chinatownTest));

      //use gini
      options = new String[2];
      options[0] = "-S";
      options[1] = "g";
      id3.setOptions(options);
      id3.buildClassifier(chinatownTrain);
      System.out.println("Id3 using measure " + id3.getAtt() + " on Chinatown Problem has test accuracy = "
              + WekaTools.accuracy(id3, chinatownTest));

      //use chisquared
      options = new String[2];
      options[0] = "-S";
      options[1] = "c";
      id3.setOptions(options);
      id3.buildClassifier(chinatownTrain);
      System.out.println("Id3 using measure " + id3.getAtt() + " on Chinatown Problem has test accuracy = "
              + WekaTools.accuracy(id3, chinatownTest));

      //use chisquared yates
      options = new String[2];
      options[0] = "-S";
      options[1] = "y";
      id3.setOptions(options);
      id3.buildClassifier(chinatownTrain);
      System.out.println("Id3 using measure " + id3.getAtt() + " with Yates on Chinatown Problem has test accuracy = "
              + WekaTools.accuracy(id3, chinatownTest));

    }
    catch (Exception e){
      e.printStackTrace();
    }
  }
}
