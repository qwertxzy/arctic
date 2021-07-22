package de.tu_darmstadt.rs.synbio.mapping.search;

import de.tu_darmstadt.rs.synbio.mapping.Assignment;
import de.tu_darmstadt.rs.synbio.simulation.SimulatorInterface;

import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

public class GeneticSearchWorker implements Callable<Void> {

  private final SimulatorInterface simulator;
  private final List<Assignment> population;
  private final ConcurrentHashMap<Assignment, Double> fitnessLookup;
  private final AtomicLong simCount;

  public GeneticSearchWorker(SimulatorInterface simulator, List<Assignment> population, ConcurrentHashMap<Assignment, Double> fitnessLookup, AtomicLong simCount) {
    this.population = population;
    this.fitnessLookup = fitnessLookup;
    this.simulator = simulator;
    this.simCount = simCount;
  }

  @Override
  public Void call() {

    for (Assignment a : population) {
      if (!fitnessLookup.containsKey(a)) {
        if (a.isValid()) {
          fitnessLookup.put(a, simulator.simulate(a));
          simCount.getAndIncrement();
        } else {
          fitnessLookup.put(a, 0.0);
        }
      }
    }

    return null;
  }
}
