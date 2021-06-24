package de.tu_darmstadt.rs.synbio.mapping;

import de.tu_darmstadt.rs.synbio.common.library.GateRealization;
import de.tu_darmstadt.rs.synbio.common.circuit.LogicGate;

import java.util.*;

public class Assignment {

    private final Map<LogicGate, GateRealization> map;

    public Assignment() {
        this.map = new HashMap<>();
    }

    public Assignment(Assignment assignment) {
        this.map = new HashMap<>(assignment.map);
    }

    public void put(LogicGate circuitGate, GateRealization realization) {
        map.put(circuitGate, realization);
    }

    public Set<LogicGate> keySet() {
        return map.keySet();
    }

    public GateRealization get(LogicGate circuitGate) {
        return map.get(circuitGate);
    }

    public Collection<GateRealization> values() {
        return map.values();
    }

    @Override
    public boolean equals(Object o) {

        if (o == this)
            return true;

        if (!(o instanceof Assignment))
            return false;

        return map.equals(((Assignment) o).map);
    }

    @Override
    public int hashCode() {
        return map.hashCode();
    }

    public Map<String, String> getIdentifierMap() {

        Map<String, String> stringMap = new HashMap<>();

        for (LogicGate circuitGate : map.keySet()) {
            stringMap.put(circuitGate.getIdentifier(), map.get(circuitGate).getIdentifier());
        }

        return stringMap;
    }

    public boolean isValid() {

        // check gate instance redundancy
        List<GateRealization> realizationList = new ArrayList<>(map.values());
        Set<GateRealization> realizationSet = new HashSet<>(map.values());

        if (realizationList.size() != realizationSet.size()) {
            return false;
        }

        // check group constraints
        List<String> usedGroups = new ArrayList<>();

        for (GateRealization realization : map.values()) {
            if (usedGroups.contains(realization.getGroup())) {
                return false;
            } else {
                usedGroups.add(realization.getGroup());
            }
        }

        return true;
    }
}
