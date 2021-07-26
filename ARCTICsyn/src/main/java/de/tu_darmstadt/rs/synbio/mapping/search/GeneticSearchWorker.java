package de.tu_darmstadt.rs.synbio.mapping.search;
import de.tu_darmstadt.rs.synbio.simulation.SimulatorInterface;

import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.atomic.AtomicLong;

public class GeneticSearchWorker implements Callable<Void> {

  private final SimulatorInterface simulator;
  private final List<GeneticSearch.Individual> population;
  private final AtomicLong simCount;
  private final AtomicLong invalidCount;

  public GeneticSearchWorker(SimulatorInterface simulator, List<GeneticSearch.Individual> population, AtomicLong simCount, AtomicLong invalidCount) {
    this.population = population;
    this.simulator = simulator;
    this.simCount = simCount;
    this.invalidCount = invalidCount;
  }

  @Override
  public Void call() {

    for (GeneticSearch.Individual individual : population) {
      if (individual.getAssignment().isValid()) {
        individual.setScore(simulator.simulate(individual.getAssignment()));
        simCount.getAndIncrement();
      } else {
        individual.setScore(0.001);
        invalidCount.getAndIncrement();
      }
    }

    return null;
  }
}
