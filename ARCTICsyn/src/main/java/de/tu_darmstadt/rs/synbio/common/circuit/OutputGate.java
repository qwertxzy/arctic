package de.tu_darmstadt.rs.synbio.common.circuit;

import de.tu_darmstadt.rs.synbio.synthesis.util.ExpressionParser;
import org.logicng.formulas.Formula;

public class OutputGate extends Gate {

    private static final Formula outputExpression = ExpressionParser.parse("x");

    public OutputGate(String identifier) {
        super(identifier, Type.OUTPUT);
    }

    @Override
    public Formula getExpression() {
        return outputExpression;
    }

    @Override
    public int getWeight() {
        return 0;
    }
}
