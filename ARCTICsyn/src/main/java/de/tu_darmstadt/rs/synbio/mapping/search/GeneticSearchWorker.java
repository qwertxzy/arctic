package de.tu_darmstadt.rs.synbio.mapping.search;
import de.tu_darmstadt.rs.synbio.common.LogicType;
import de.tu_darmstadt.rs.synbio.common.circuit.LogicGate;
import de.tu_darmstadt.rs.synbio.common.library.GateRealization;
import de.tu_darmstadt.rs.synbio.mapping.Assignment;
import de.tu_darmstadt.rs.synbio.mapping.util.BitField;
import de.tu_darmstadt.rs.synbio.simulation.SimulatorInterface;

import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.atomic.AtomicLong;

public class GeneticSearchWorker implements Callable<Void> {

  private final SimulatorInterface simulator;
  private final List<GeneticSearchIndividual> population;
  private final Map<LogicType, List<BitField>> geneEncoding;
  private final Map<LogicType, List<GateRealization>> realizations;
  private final Set<LogicGate> assignmentGates;
  private final AtomicLong simCount;
  private final AtomicLong invalidCount;

  public GeneticSearchWorker(SimulatorInterface simulator, List<GeneticSearchIndividual> population, Map<LogicType,
                              List<GateRealization>> realizations, Map<LogicType, List<BitField>> geneEncoding,
                              Set<LogicGate> assignmentGates, AtomicLong simCount, AtomicLong invalidCount) {
    this.population = population;
    this.simulator = simulator;
    this.geneEncoding = geneEncoding;
    this.realizations = realizations;
    this.assignmentGates = assignmentGates;
    this.simCount = simCount;
    this.invalidCount = invalidCount;
  }

  @Override
  public Void call() {

    for (GeneticSearchIndividual individual : population) {
      Assignment assignment = GeneticSearchIndividual.geneticDecode(
          realizations, geneEncoding, individual.getEncodedAssignment(), assignmentGates);

      // Assignment might be null if a genome contained an out-of-bounds index
      if (assignment != null && assignment.isValid()) {
        individual.setScore(simulator.simulate(assignment));

        simCount.getAndIncrement();
      } else {
        individual.setScore(0.001);

        invalidCount.getAndIncrement();
      }
    }

    return null;
  }
}
