package ml_6002b_coursework;

import weka.core.Instance;

import java.util.HashMap;

public interface VotingSystem {
    //apply voting system from votes map
    public double countVotes(TreeEnsemble c, Instance instance) throws Exception;
}
