package de.tu_darmstadt.rs.synbio.common.circuit;

import org.logicng.formulas.Formula;

public class InputGate extends Gate {

    private final Formula inputExpression;

    public InputGate(Formula inputExpression, String identifier) {
        super(identifier, Type.INPUT);
        this.inputExpression = inputExpression;
    }

    @Override
    public Formula getExpression() {
        return inputExpression;
    }

    @Override
    public int getWeight() {
        return 0;
    }
}
