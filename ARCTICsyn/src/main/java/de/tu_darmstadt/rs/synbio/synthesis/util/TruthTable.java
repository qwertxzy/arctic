package de.tu_darmstadt.rs.synbio.synthesis.util;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonValue;
import org.apache.commons.lang3.builder.EqualsBuilder;
import org.logicng.datastructures.Assignment;
import org.logicng.formulas.Formula;
import org.logicng.formulas.Literal;
import org.logicng.formulas.Variable;

import java.util.*;

public class TruthTable {

    private Integer truthTable;
    private Integer firstUnusedBit;

    public TruthTable(Formula f) {

        this.truthTable = 0;
        this.firstUnusedBit = 0;

        for (Assignment assignment : getAllAssignments(f.variables())) {
            this.insert(f.evaluate(assignment));
        }
    }

    public Integer getTruthTable() {
        return truthTable;
    }

    public int getSupportSize() {
        int supportSize;

        switch (firstUnusedBit) {
            case 2:
                supportSize = 1;
                break;
            case 4:
                supportSize = 2;
                break;
            case 8:
                supportSize = 3;
                break;
            case 16:
                supportSize = 4;
                break;
            case 32:
                supportSize = 5;
                break;
            default:
                supportSize = 0;
        }

        return supportSize;
    }

    private Integer getLength() {
        return firstUnusedBit;
    }

    private void insert(boolean value) throws IndexOutOfBoundsException {

        if (firstUnusedBit > 32)
            throw new IndexOutOfBoundsException("Truth table may maximally contain 32 bits / 5 inputs.");

        truthTable |= (value ? 1 : 0) << firstUnusedBit;
        firstUnusedBit++;
    }

    private LinkedList<Assignment> getAllAssignments(SortedSet<Variable> variables) {

        LinkedList<Assignment> assignments = new LinkedList<>();
        NavigableSet<Variable> variableSet = new TreeSet<>(variables);
        int pos;

        // iterate over all possible assignments
        for (int i = 0; i < (1 << variables.size()); i ++) {

            LinkedList<Literal> literals = new LinkedList<>();
            pos = 0;

            // iterate over all variables per assignment in descending order (like in abc)
            Iterator<Variable> variableIterator = variableSet.descendingIterator();
            while (variableIterator.hasNext()) {

                Variable var = variableIterator.next();

                // add non-negated or negated variable to assigment
                boolean positiveVar = (i & (1 << pos)) != 0;
                literals.add(positiveVar ? var : var.negate());
                pos ++;

            }
            assignments.add(new Assignment(literals));
        }
        return assignments;
    }

    public String toString() {

        StringBuilder builder = new StringBuilder();

        for (int i = 0; i < firstUnusedBit; i ++) {
            builder.insert(0, ((truthTable & (1 << i)) != 0) ? "1" : "0");
        }

        return builder.toString();
    }

    public boolean equalsLogically(TruthTable cmp) {
        return new EqualsBuilder().append(this.truthTable, cmp.getTruthTable()).append(this.getLength(), cmp.getLength()).isEquals();
    }

    @JsonValue
    public String toJson() {
        return this.truthTable + "," + this.firstUnusedBit;
    }

    @JsonCreator
    public TruthTable(String jsonValue) {

        String[] values = jsonValue.split(",");
        this.truthTable = Integer.parseInt(values[0], 10);
        this.firstUnusedBit = Integer.parseInt(values[1], 10);
    }
}
