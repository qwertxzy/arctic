package de.tu_darmstadt.rs.synbio.simulation;

import de.tu_darmstadt.rs.synbio.mapping.Assignment;
import de.tu_darmstadt.rs.synbio.common.circuit.Circuit;

public class SimulationResult implements Comparable<SimulationResult> {

    private final Circuit structure;
    private final Assignment assignment;
    private final double score;

    private long neededSimulations = 0;

    public SimulationResult(Circuit structure, Assignment assignment, double score) {
        this.structure = structure;
        this.assignment = assignment;
        this.score = score;
    }

    public double getScore() {
        return score;
    }

    public Circuit getStructure() {
        return structure;
    }

    public Assignment getAssignment() {
        return assignment;
    }

    public void setNeededSimulations(long sims) {
        this.neededSimulations = sims;
    }

    public long getNeededSimulations() {
        return neededSimulations;
    }

    @Override
    public int compareTo(SimulationResult cmp) {
        if (this.score < cmp.score)
            return -1;
        else if (cmp.score < this.score)
            return 1;
        return 0;
    }
}
