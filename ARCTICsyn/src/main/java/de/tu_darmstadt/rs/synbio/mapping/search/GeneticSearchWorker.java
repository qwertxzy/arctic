package de.tu_darmstadt.rs.synbio.mapping.search;

import de.tu_darmstadt.rs.synbio.common.circuit.Circuit;
import de.tu_darmstadt.rs.synbio.common.library.GateLibrary;
import de.tu_darmstadt.rs.synbio.mapping.Assignment;
import de.tu_darmstadt.rs.synbio.simulation.SimulationConfiguration;
import de.tu_darmstadt.rs.synbio.simulation.SimulatorInterface;

import java.util.HashMap;
import java.util.List;
import java.util.concurrent.Callable;

public class GeneticSearchWorker implements Callable<HashMap<Assignment, Double>> {

  private final SimulatorInterface simulator;
  private final Circuit structure;
  private final List<Assignment> population;

  public GeneticSearchWorker(SimulationConfiguration simConfig, GateLibrary gateLibrary, Circuit structure, List<Assignment> population) {
    this.population = population;
    this.structure = structure;
    this.simulator = new SimulatorInterface(simConfig, gateLibrary.getSourceFile());
  }

  @Override
  public HashMap<Assignment, Double> call() {
    HashMap<Assignment, Double> fitnessLookup = new HashMap<>();

    simulator.initSimulation(structure);

    for (Assignment a : population) {
      fitnessLookup.put(a, simulator.simulate(a));
    }

    simulator.shutdown();

    return fitnessLookup;
  }
}
