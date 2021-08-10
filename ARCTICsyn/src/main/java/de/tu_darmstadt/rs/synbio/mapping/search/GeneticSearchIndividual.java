package de.tu_darmstadt.rs.synbio.mapping.search;

import de.tu_darmstadt.rs.synbio.common.LogicType;
import de.tu_darmstadt.rs.synbio.common.circuit.LogicGate;
import de.tu_darmstadt.rs.synbio.common.library.GateRealization;
import de.tu_darmstadt.rs.synbio.mapping.Assignment;

import java.util.List;
import java.util.Map;
import java.util.Set;

public class GeneticSearchIndividual {
    private Assignment assignment;
    private String encodedAssignment;
    private Double score;


    GeneticSearchIndividual(Assignment assignment) {
        this.assignment = assignment;
        this.encodedAssignment = null;
        this.score = 0.0;
    }
    public String getEncodedAssignment() {
        return encodedAssignment;
    }

    public void setEncodedAssignment(String encodedAssignment) {
        this.encodedAssignment = encodedAssignment;
    }
    public Assignment getAssignment() {
        return this.assignment;
    }

    public void setAssignment(Assignment assignment) {
        this.assignment = assignment;
    }
    public void setScore(Double score) {
        this.score = score;
    }

    public Double getScore() {
        return score;
    }

    public static String geneticEncode(Map<LogicType, List<GateRealization>> realizations, Map<LogicType, List<String>> geneEncoding, Assignment assignment) {
        StringBuilder encodedAssignment = new StringBuilder();

        for (LogicGate gate : assignment.keySet()) {
            LogicType gateType = gate.getLogicType();
            int listIndex = realizations.get(gateType).indexOf(assignment.get(gate));
            encodedAssignment.append(geneEncoding.get(gateType).get(listIndex));
        }

        return encodedAssignment.toString();
    }
    public static Assignment geneticDecode(Map<LogicType, List<GateRealization>> realizations, Map<LogicType, List<String>> geneEncoding, String encodedAssignment, Set<LogicGate> assignmentGates) {
        Assignment assignment = new Assignment();

        int i = 0;
        while (encodedAssignment.length() > 0) {
            LogicGate gate = assignmentGates.toArray(new LogicGate[0])[i];
            int bitWidth = geneEncoding.get(gate.getLogicType()).get(0).length();
            String encodedGate = encodedAssignment.substring(0, bitWidth);
            encodedAssignment = encodedAssignment.substring(bitWidth);
            int realizationIndex = geneEncoding.get(gate.getLogicType()).indexOf(encodedGate); // TODO: indexOf is inefficient, index could be calculated
            GateRealization realization;
            try {
                realization = realizations.get(gate.getLogicType()).get(realizationIndex);
            } catch (IndexOutOfBoundsException _e) {
                return null; // Invalid encoding -> return null to signal this
            }
            assignment.put(gate, realization);
            i++;
        }
        return assignment;
    }
}
