package de.tu_darmstadt.rs.synbio.mapping.search;

import de.tu_darmstadt.rs.synbio.common.circuit.Circuit;
import de.tu_darmstadt.rs.synbio.mapping.MappingConfiguration;
import de.tu_darmstadt.rs.synbio.simulation.SimulationConfiguration;
import de.tu_darmstadt.rs.synbio.common.library.GateLibrary;
import de.tu_darmstadt.rs.synbio.simulation.SimulationResult;

public abstract class AssignmentSearchAlgorithm {

    protected Circuit structure;
    protected GateLibrary gateLib;
    protected SimulationConfiguration simConfig;
    protected MappingConfiguration mapConfig;

    public AssignmentSearchAlgorithm(Circuit structure, GateLibrary lib, MappingConfiguration mapConfig, SimulationConfiguration simConfig) {
        this.structure = structure;
        this.gateLib = lib;
        this.mapConfig = mapConfig;
        this.simConfig = simConfig;
    }

    public abstract SimulationResult assign();
}
