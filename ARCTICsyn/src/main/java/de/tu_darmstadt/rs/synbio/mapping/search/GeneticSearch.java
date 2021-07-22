package de.tu_darmstadt.rs.synbio.mapping.search;

import de.tu_darmstadt.rs.synbio.common.LogicType;
import de.tu_darmstadt.rs.synbio.common.circuit.Circuit;
import de.tu_darmstadt.rs.synbio.common.circuit.LogicGate;
import de.tu_darmstadt.rs.synbio.common.library.GateLibrary;
import de.tu_darmstadt.rs.synbio.common.library.GateRealization;
import de.tu_darmstadt.rs.synbio.mapping.Assignment;
import de.tu_darmstadt.rs.synbio.mapping.MappingConfiguration;
import de.tu_darmstadt.rs.synbio.mapping.assigner.RandomAssigner;
import de.tu_darmstadt.rs.synbio.simulation.SimulationConfiguration;
import de.tu_darmstadt.rs.synbio.simulation.SimulationResult;
import de.tu_darmstadt.rs.synbio.simulation.SimulatorInterface;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicLong;

public class GeneticSearch extends AssignmentSearchAlgorithm {

  private static final Logger logger = LoggerFactory.getLogger(GeneticSearch.class);

  private final int populationSize;
  private final int eliteNumber ;
  private final int crossoverCount;
  private final int iterationCount;
  private final double mutationRate;

  public GeneticSearch(Circuit structure, GateLibrary lib, MappingConfiguration mapConfig, SimulationConfiguration simConfig) {
    super(structure, lib, mapConfig, simConfig);

    this.populationSize = mapConfig.getPopulationSize();
    this.eliteNumber = mapConfig.getEliteNumber();
    this.crossoverCount = mapConfig.getCrossoverCount();
    this.iterationCount = mapConfig.getIterationCount();
    this.mutationRate = mapConfig.getMutationRate();
  }

  public SimulationResult assign() {

    // Initialize gate library & simulators

    HashMap<LogicType, List<GateRealization>> realizations = gateLib.getRealizations();

    int maxThreads = Runtime.getRuntime().availableProcessors() - 1;
    int availableProcessors = simConfig.simLimitThreads() ? Math.min(simConfig.getSimLimitThreadsNum(), maxThreads) : maxThreads;

    List<SimulatorInterface> simulators = new ArrayList<>();
    for (int i = 0; i < availableProcessors; i++) {
      SimulatorInterface sim = new SimulatorInterface(simConfig, gateLib.getSourceFile());
      sim.initSimulation(structure);
      simulators.add(sim);
    }

    RandomAssigner randomAssigner = new RandomAssigner(gateLib, structure);
    Random random = new Random();

    // Declare population related variables

    List<Assignment> currentPopulation = new ArrayList<>();
    List<Assignment> nextPopulation = new ArrayList<>();
    ConcurrentHashMap<Assignment, Double> fitnessLookup = new ConcurrentHashMap<>();
    Assignment bestAssignment = null;

    // Generate initial population

    for (int i = 0; i < populationSize; i++) {
      currentPopulation.add(randomAssigner.getNextAssignment());
    }

    // Loop until exit condition is met

    currentIteration = 0;
    AtomicLong simCount = new AtomicLong(); // Atomic long to increment it from lambda expr

    while (checkExitCondition()) {
      logger.info("GEN: " + currentIteration);

      // Calculate fitness of current population

      List<GeneticSearchWorker> workers = new ArrayList<>();
      int sliceLength = (int) Math.ceil(currentPopulation.size() / availableProcessors);

      // Split up current population into even slices to simulate
      for (int i = 0; i < availableProcessors; i++) {
        List<Assignment> slice = currentPopulation.subList(i * sliceLength, (i == availableProcessors - 1 ? currentPopulation.size() : (i + 1) * sliceLength));
        workers.add(new GeneticSearchWorker(simulators.get(i), slice, fitnessLookup, simCount));
      }

      ExecutorService executor = Executors.newFixedThreadPool(availableProcessors);

      try {
        executor.invokeAll(workers);
      } catch (InterruptedException e) {
        logger.error(e.getMessage());
      }

      // Now sort the current population based on the new fitness data

      if (mapConfig.getOptimizationType().equals(MappingConfiguration.OptimizationType.MAXIMIZE)) {
        // Sort descending
        currentPopulation.sort((a1, a2) -> fitnessLookup.get(a1).compareTo(fitnessLookup.get(a2)) * (-1));
      } else {
        // Sort ascending
        currentPopulation.sort(Comparator.comparing(fitnessLookup::get));
      }

      logger.info("|TOP: " + fitnessLookup.get(currentPopulation.get(0)));
      logger.info("|TOP5: " + ( currentPopulation.subList(0, 5).stream().map(fitnessLookup::get).reduce(0.0, Double::sum) / 5 )); // TODO: doesn't work
      logger.info("|AVG: " + ( currentPopulation.stream().map(fitnessLookup::get).reduce(0.0, Double::sum) / populationSize ));

      // Select the best n as elites to be carried over to the next generation

      for (int i = 0; i < eliteNumber; i++) {
        nextPopulation.add(currentPopulation.get(i));
      }

      // Take n best individuals, generate n / 2 crossed over children

      for (int i = 0; i < crossoverCount; i += 2) {
        Assignment firstParent = currentPopulation.get(i);
        Assignment secondParent = currentPopulation.get(i + 1);
        Assignment firstChild = new Assignment();
        Assignment secondChild = new Assignment();

        int crossoverPoint = random.nextInt(firstParent.keySet().size());
        LogicGate[] genomeKeys = firstParent.keySet().toArray(new LogicGate[0]);

        for (int j = 0; j < genomeKeys.length; j++) {
          GateRealization firstRealization;
          GateRealization secondRealization;

          if (crossoverPoint < j) {
            firstRealization = firstParent.get(genomeKeys[j]);
            secondRealization = secondParent.get(genomeKeys[j]);
          } else {
            firstRealization = secondParent.get(genomeKeys[j]);
            secondRealization = firstParent.get(genomeKeys[j]);
          }
          firstChild.put(genomeKeys[j], firstRealization);
          secondChild.put(genomeKeys[j], secondRealization);
        }

        nextPopulation.add(firstChild);
        nextPopulation.add(secondChild);
      }

      // Apply random mutation (select a random equivalent gate implementation for random individuals)

      for (Assignment individual : currentPopulation) {
        if (random.nextDouble() <= mutationRate) {
          LogicGate[] individualGates = individual.keySet().toArray(new LogicGate[0]);
          LogicGate mutatingGate = individualGates[random.nextInt(individualGates.length)];

          List<GateRealization> possibleAlternatives = realizations.get(mutatingGate.getLogicType());
          GateRealization newRealization = possibleAlternatives.get(random.nextInt(possibleAlternatives.size()));

          individual.put(mutatingGate, newRealization);
          nextPopulation.add(individual);
        }
      }

      // Fill up the rest with new random individuals

      for (int i = 0; i < populationSize - nextPopulation.size(); i++) {
        nextPopulation.add(randomAssigner.getNextAssignment());
      }

      // Move the next generation's individuals to the current generation
      // TODO: optimize this step, maybe work on two alternating lists instead of copying

      bestAssignment = currentPopulation.get(0);
      Collections.copy(currentPopulation, nextPopulation);
      nextPopulation.clear();
    }

    SimulationResult result = new SimulationResult(structure, bestAssignment, fitnessLookup.get(bestAssignment));
    result.setNeededSimulations(simCount.get());

    for (SimulatorInterface sim : simulators) {
      sim.shutdown();
    }

    return result;
  }

  private int currentIteration;

  private boolean checkExitCondition() {
    currentIteration++;
    // TODO: Switch depending on exit after n iterations or achieved score or whatever
    return currentIteration <= iterationCount;
  }
}
