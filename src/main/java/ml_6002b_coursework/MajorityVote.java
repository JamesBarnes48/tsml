package ml_6002b_coursework;

import weka.core.Instance;

import java.util.HashMap;
import java.util.Iterator;
import java.util.Map;

public class MajorityVote implements VotingSystem {
    @Override
    public double countVotes(TreeEnsemble c, Instance instance) throws Exception {
        //poll classifiers for votes
        //HashMap<Double, Integer> votes = c.pollClassifiers(instance);
        //select class with most votes
        double mostVoted = 0;
        //Iterator it = votes.entrySet().iterator();
        //while (it.hasNext()) {
            //Map.Entry<Double, Integer> entry = (Map.Entry<Double, Integer>) it.next();
            //if (entry.getValue() > mostVoted) {
            //    mostVoted = entry.getKey();
            //}
        //}
        /*
        System.out.println("Predicting instance");
        votes.entrySet().forEach(entry -> {
            System.out.println(entry.getKey() + " " + entry.getValue());
        });*/
        return mostVoted;
    }
}
