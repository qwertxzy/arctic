package de.tu_darmstadt.rs.synbio.common.circuit;

import de.tu_darmstadt.rs.synbio.common.LogicType;
import de.tu_darmstadt.rs.synbio.synthesis.util.TruthTable;
import de.tu_darmstadt.rs.synbio.synthesis.util.ExpressionParser;
import org.jgrapht.io.Attribute;
import org.jgrapht.io.VertexProvider;
import org.logicng.formulas.Formula;

import java.util.Map;

public abstract class Gate {

    private final String identifier;
    private final Type type;

    public enum Type {
        INPUT,
        LOGIC,
        OUTPUT
    };

    public Gate(String identifier, Gate.Type type) {
        this.identifier = identifier;
        this.type = type;
    }

    public abstract Formula getExpression();

    public TruthTable getTruthTable() {
        return new TruthTable(getExpression());
    }

    public String getIdentifier() {
        return identifier;
    }

    public abstract int getWeight();

    public Type getType() {
        return type;
    }

    static class GateProvider implements VertexProvider<Gate> {

        @Override
        public Gate buildVertex(String s, Map<String, Attribute> map) {

            Formula expression = ExpressionParser.parse(map.get("expression").getValue());
            String primitiveIdentifier = map.get("primitiveIdentifier").getValue();

            switch(map.get("type").getValue()) {
                case "INPUT":
                    return new InputGate(expression, s);
                case "OUTPUT":
                    return new OutputGate(s);
                default:
                    return new LogicGate(s, LogicType.valueOf(primitiveIdentifier));
            }
        }
    }
}
