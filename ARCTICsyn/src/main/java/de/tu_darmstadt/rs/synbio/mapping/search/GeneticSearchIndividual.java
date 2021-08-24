package de.tu_darmstadt.rs.synbio.mapping.search;

import de.tu_darmstadt.rs.synbio.common.LogicType;
import de.tu_darmstadt.rs.synbio.common.circuit.LogicGate;
import de.tu_darmstadt.rs.synbio.common.library.GateRealization;
import de.tu_darmstadt.rs.synbio.mapping.Assignment;
import de.tu_darmstadt.rs.synbio.mapping.util.BitField;

import java.util.List;
import java.util.Map;
import java.util.Set;

public class GeneticSearchIndividual implements Comparable<GeneticSearchIndividual> {
    private BitField encodedAssignment;
    private Double score;

    GeneticSearchIndividual(BitField encodedAssignment) {
        this.encodedAssignment = encodedAssignment;
        this.score = 0.0;
    }

    GeneticSearchIndividual(GeneticSearchIndividual other) {
        this.encodedAssignment = new BitField(other.getEncodedAssignment());
        this.score = other.getScore();
    }

    public BitField getEncodedAssignment() {
        return encodedAssignment;
    }

    public void setEncodedAssignment(BitField encodedAssignment) {
        this.encodedAssignment = encodedAssignment;
    }

    public void setScore(Double score) {
        this.score = score;
    }

    public Double getScore() {
        return score;
    }

    public static BitField geneticEncode(Map<LogicType, List<GateRealization>> realizations, Map<LogicType,
        List<BitField>> geneEncoding, Assignment assignment) {

        BitField encodedAssignment = new BitField();

        for (LogicGate gate : assignment.keySet()) {
            LogicType gateType = gate.getLogicType();
            int listIndex = realizations.get(gateType).indexOf(assignment.get(gate));

            BitField encodedGate = geneEncoding.get(gateType).get(listIndex);

            encodedAssignment.append(encodedGate);
        }

        return encodedAssignment;
    }
    public static Assignment geneticDecode(Map<LogicType, List<GateRealization>> realizations, Map<LogicType,
        List<BitField>> geneEncoding, BitField encodedAssignment, Set<LogicGate> assignmentGates) {

        Assignment assignment = new Assignment();

        int i = 0;
        while (encodedAssignment.length() > 0) {
            LogicGate gate = assignmentGates.toArray(new LogicGate[0])[i];
            int bitWidth = geneEncoding.get(gate.getLogicType()).get(0).length();
            BitField encodedGate = encodedAssignment.subfield(0, bitWidth);
            encodedAssignment = encodedAssignment.subfield(bitWidth, encodedAssignment.length());
            int realizationIndex = geneEncoding.get(gate.getLogicType()).indexOf(encodedGate);
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

    @Override
    public int compareTo(GeneticSearchIndividual other) {
        return this.getScore().compareTo(other.getScore()) * -1;
    }

}
