package distances.derivative_time_domain.ddtw;

import distances.time_domain.dtw.Dtw;
import evaluation.tuning.ParameterSpace;
import timeseriesweka.filters.DerivativeFilter;
import weka.core.Instance;
import weka.core.Instances;
import weka.filters.Filter;

public class Ddtw extends Dtw {

    private final Filter derivative = new DerivativeFilter();

    @Override
    public double distance(Instance a,
                           Instance b,
                           final double cutOff) {
        Instances instances = new Instances(a.dataset(), 0);
        instances.add(a);
        instances.add(b);
        try {
            instances = Filter.useFilter(instances, derivative);
            a = instances.get(0);
            b = instances.get(1);
            return super.distance(a, b, cutOff);
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }

    public static final String NAME = "DDTW";

    @Override
    public String toString() {
        return NAME;
    }

    public static ParameterSpace discreteParameterSpace(Instances instances) {
        ParameterSpace parameterSpace = Dtw.discreteParameterSpace(instances);
        parameterSpace.addParameter(DISTANCE_MEASURE_KEY, new String[] {NAME});
        return parameterSpace;
    }

    public static ParameterSpace euclideanParameterSpace() {
        ParameterSpace parameterSpace = Dtw.euclideanParameterSpace();
        parameterSpace.addParameter(DISTANCE_MEASURE_KEY, new String[] {NAME});
        return parameterSpace;
    }

    public static ParameterSpace fullWindowParameterSpace() {
        ParameterSpace parameterSpace = Dtw.fullWindowParameterSpace();
        parameterSpace.addParameter(DISTANCE_MEASURE_KEY, new String[] {NAME});
        return parameterSpace;
    }
}