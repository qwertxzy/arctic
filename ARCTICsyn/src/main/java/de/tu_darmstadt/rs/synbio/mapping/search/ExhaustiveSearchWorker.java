package de.tu_darmstadt.rs.synbio.mapping.search;

import de.tu_darmstadt.rs.synbio.mapping.MappingConfiguration;
import de.tu_darmstadt.rs.synbio.simulation.SimulationConfiguration;
import de.tu_darmstadt.rs.synbio.mapping.Assignment;
import de.tu_darmstadt.rs.synbio.common.circuit.Circuit;
import de.tu_darmstadt.rs.synbio.mapping.assigner.ExhaustiveAssigner;
import de.tu_darmstadt.rs.synbio.common.library.GateLibrary;
import de.tu_darmstadt.rs.synbio.simulation.SimulationResult;
import de.tu_darmstadt.rs.synbio.simulation.SimulatorInterface;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.Callable;

public class ExhaustiveSearchWorker implements Callable<SimulationResult> {

    private static final Logger logger = LoggerFactory.getLogger(ExhaustiveSearchWorker.class);

    private final SimulatorInterface simulator;
    private final ExhaustiveAssigner assigner;
    private final Circuit structure;
    private final SimulationConfiguration simConfig;
    private final MappingConfiguration mapConfig;
    private final ExhaustiveSearch.ExhaustiveLogger exhaustiveLogger;

    public ExhaustiveSearchWorker(ExhaustiveSearch.ExhaustiveLogger exhaustiveLogger, ExhaustiveAssigner assigner, Circuit structure, MappingConfiguration mapConfig,
                                  SimulationConfiguration simConfig, GateLibrary gateLibrary) {
        this.mapConfig = mapConfig;
        this.simConfig = simConfig;
        this.simulator = new SimulatorInterface(simConfig, gateLibrary.getSourceFile());
        this.assigner = assigner;
        this.structure = structure;
        this.exhaustiveLogger = exhaustiveLogger;
    }

    @Override
    public SimulationResult call() {

        simulator.initSimulation(structure);

        Assignment assignment = assigner.getNextAssignment();
        SimulationResult bestRes = null;

        while (assignment != null && !Thread.interrupted()) {
            SimulationResult result = new SimulationResult(structure, assignment, simulator.simulate(assignment));

            exhaustiveLogger.append("" + result.getScore());

            if (bestRes == null || (mapConfig.getOptimizationType().compare(bestRes.getScore(), result.getScore()))) {
                bestRes = result;
            }

            assignment = assigner.getNextAssignment();
        }

        simulator.shutdown();

        return bestRes;
    }
}
