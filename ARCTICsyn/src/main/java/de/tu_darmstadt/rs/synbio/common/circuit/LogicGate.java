package de.tu_darmstadt.rs.synbio.common.circuit;

import de.tu_darmstadt.rs.synbio.common.LogicType;
import org.logicng.formulas.Formula;

public class LogicGate extends Gate {

    private final LogicType logicType;

    public LogicGate(String identifier, LogicType logicType) {
        super(identifier, Type.LOGIC);
        this.logicType = logicType;
    }

    public LogicType getLogicType() {
        return logicType;
    }

    @Override
    public Formula getExpression() {
        return logicType.getExpression();
    }

    @Override
    public int getWeight() {
        return logicType.getWeight();
    }
}
