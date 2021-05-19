package ml_6002b_coursework;

import org.w3c.dom.Attr;
import weka.core.*;
import weka.filters.Filter;
import weka.filters.supervised.attribute.Discretize;

import java.util.*;

/**
 * Interface for alternative attribute split measures for Part 2.2 of the coursework
 *
 */
public interface AttributeSplitMeasure {

    double computeAttributeQuality(Instances data, Attribute att) throws Exception;

    /**
     * Splits a dataset according to the values of a nominal attribute.
     *
     * @param data the data which is to be split
     * @param att the attribute to be used for splitting
     * @return the sets of instances produced by the split
     */
     default Instances[] splitData(Instances data, Attribute att) {
         //if numeric split on numeric instead
         if(att.isNumeric()) {
             return splitDataOnNumeric(data,att).getKey();
         }
        Instances[] splitData = new Instances[att.numValues()];
        for (int j = 0; j < att.numValues(); j++) {
            splitData[j] = new Instances(data, data.numInstances());
        }
        Enumeration instEnum = data.enumerateInstances();
        while (instEnum.hasMoreElements()) {
            Instance inst = (Instance) instEnum.nextElement();
            splitData[(int) inst.value(att)].add(inst);
        }
        for (int i = 0; i < splitData.length; i++) {
            splitData[i].compactify();
        }
        return splitData;
    }

    default Map.Entry<Instances[], Double> splitDataOnNumeric(Instances data, Attribute att){
         //array of instances used to hold the two parts of the split
        Instances[] splitData = new Instances[2];
        splitData[0] = new Instances(data, 0);
        splitData[1] = new Instances(data, 0);

        //find random split value
        AttributeStats attStats = data.attributeStats(att.index());
        double max = attStats.numericStats.max;
        double min = attStats.numericStats.min;
        double random = ((Math.random() * (max - min)) + min);

        //bin instances based on random split value
        for(int i = 0; i < data.numInstances(); i++)
        {
            Instance checkedIns = data.instance(i);
            if (checkedIns.value(att) > random) {
                splitData[0].add(checkedIns);
            }
            else {
                splitData[1].add(checkedIns);
            }
        }
        //compactify split data
        for (Instances ins : splitData) {
            ins.compactify();
        }

        Map<Instances[], Double>
                result = Collections
                .singletonMap(splitData, random);

        Map.Entry<Instances[], Double> entry = new AbstractMap.SimpleEntry<>(splitData, random);

        return entry;

    }

}
