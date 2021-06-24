package de.tu_darmstadt.rs.synbio.common.circuit;

import org.jgrapht.graph.DefaultEdge;
import org.jgrapht.io.Attribute;
import org.jgrapht.io.EdgeProvider;
import org.logicng.formulas.FormulaFactory;
import org.logicng.formulas.Variable;

import java.util.Map;

public class Wire extends DefaultEdge {

    private final Variable variable;

    public Wire(Variable variable) {
        super();
        this.variable = variable;
    }

    public Variable getVariable() {
        return variable;
    }

    static class WireProvider implements EdgeProvider<Gate, Wire> {

        @Override
        public Wire buildEdge(Gate gate, Gate v1, String s, Map<String, Attribute> map) {

            FormulaFactory varFac = new FormulaFactory();
            Variable var = varFac.variable(map.get("variable").getValue());

            return new Wire(var);
        }
    }
}
