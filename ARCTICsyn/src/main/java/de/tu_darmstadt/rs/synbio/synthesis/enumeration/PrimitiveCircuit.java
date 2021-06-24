package de.tu_darmstadt.rs.synbio.synthesis.enumeration;

import java.util.ArrayList;
import java.util.List;

public class PrimitiveCircuit {

    private final List<Entry> circuit;

    public PrimitiveCircuit(int numRows) {
        this.circuit = new ArrayList<>(numRows);
    }

    public PrimitiveCircuit(PrimitiveCircuit circuit) {
        this.circuit = new ArrayList<>(circuit.circuit);
    }

    public Entry getEntry(int i) {
        return circuit.get(i);
    }

    public void insertEntry(int combId, int combIndex) {
        circuit.add(0, new Entry(combId, combIndex));
    }

    public void addEntry(int row, int combId, int combIndex) {
        circuit.add(row, new Entry(combId, combIndex));
    }

    public List<Entry> getList() {
        return circuit;
    }

    public int getDepth() {
        return circuit.size();
    }

    public static class Entry {

        public int combId;
        public int index;

        public Entry(int combId, int index) {
            this.combId = combId;
            this.index = index;
        }
    }

}
