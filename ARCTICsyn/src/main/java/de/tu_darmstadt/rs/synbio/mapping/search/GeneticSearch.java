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

    List<Individual> currentPopulation = new ArrayList<>();
    List<Individual> nextPopulation = new ArrayList<>();

    // Generate initial population

    for (int i = 0; i < populationSize; i++) {
      currentPopulation.add(new Individual(randomAssigner.getNextAssignment()));
    }
    Individual bestIndividual = currentPopulation.get(0);

    // Loop until exit condition is met

    currentIteration = 0;

    // Atomic long to increment it from simulation threads
    AtomicLong simCount = new AtomicLong(0);
    AtomicLong invalidCount = new AtomicLong();

    logger.info("Generation, Invalid, Top, Top5, Average");

    while (checkExitCondition()) {
      // Calculate fitness of current population

      List<GeneticSearchWorker> workers = new ArrayList<>();
      double sliceLength = currentPopulation.size() / (double) availableProcessors;
      invalidCount.set(0);

      // Split up current population into even slices to simulate
      for (int i = 0; i < availableProcessors; i++) {
        List<Individual> slice = currentPopulation.subList((int) Math.ceil(i * sliceLength), (int) Math.ceil((i + 1) * sliceLength));
        workers.add(new GeneticSearchWorker(simulators.get(i), slice, simCount, invalidCount));
      }

      ExecutorService executor = Executors.newFixedThreadPool(availableProcessors);

      try {
        executor.invokeAll(workers);
      } catch (InterruptedException e) {
        logger.error(e.getMessage());
      }

      // Select the best n as elites to be carried over to the next generation

      for (int i = 0; i < eliteNumber; i++) {
        nextPopulation.add(currentPopulation.get(i));
      }

      // Apply stochastic universal sampling to choose parents via roulette wheel selection

      double totalFitness = currentPopulation.stream().map(Individual::getScore).reduce(0.0, Double::sum);
      double intervalDistance = totalFitness / crossoverCount;
      double startPoint = intervalDistance * random.nextDouble();

      List<Double> pointers = new ArrayList<>();
      for (int i = 0; i < crossoverCount; i++) {
        pointers.add(startPoint + i * intervalDistance);
      }

      List<Assignment> parents = new ArrayList<>();
      for (Double point : pointers) {
        int i = 1;
        while (currentPopulation.subList(0, i).stream().map(Individual::getScore).reduce(0.0, Double::sum) < point) {
          i++;
        }
        parents.add(currentPopulation.get(i - 1).getAssignment()); // TODO: is this -1 correct? look over the whole SUS in general
      }

      // Cross parents with random crossover point

      for (int i = 0; i < parents.size(); i += 2) {
        Assignment firstParent = parents.get(i);
        Assignment secondParent = parents.get(i + 1);
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

        nextPopulation.add(new Individual(firstChild));
        nextPopulation.add(new Individual(secondChild));
      }

      // Apply random mutation (select a random equivalent gate implementation for random individuals)

      for (Individual individual : nextPopulation) {
        if (random.nextDouble() <= mutationRate) {
          Assignment individualAssignment = individual.getAssignment();
          LogicGate[] individualGates = individualAssignment.keySet().toArray(new LogicGate[0]);
          LogicGate mutatingGate = individualGates[random.nextInt(individualGates.length)];

          List<GateRealization> possibleAlternatives = realizations.get(mutatingGate.getLogicType());
          GateRealization newRealization = possibleAlternatives.get(random.nextInt(possibleAlternatives.size()));

          individual.getAssignment().put(mutatingGate, newRealization);
        }
      }

      // Fill up the rest with new random individuals

      for (int i = 0; i < populationSize - nextPopulation.size(); i++) {
        nextPopulation.add(new Individual(randomAssigner.getNextAssignment()));
      }

      // Extract the best individual
      // TODO: make this respect the OptimizationType here & above in the parent selection
      bestIndividual = currentPopulation.stream().max(Comparator.comparing(Individual::getScore)).get();

      logger.info(
          currentIteration +
              "," + invalidCount.get() +
              "," + bestIndividual.getScore() +
              "," + ( (currentPopulation.subList(0, 5).stream().map(Individual::getScore).reduce(0.0, Double::sum)) / 5.0 ) +
              "," + ( currentPopulation.stream().map(Individual::getScore).reduce(0.0, Double::sum) / populationSize )
      );

      // Move the next generation's individuals to the current generation
      // TODO: optimize this step, maybe work on two alternating lists instead of copying
      Collections.copy(currentPopulation, nextPopulation);
      nextPopulation.clear();
    }

    SimulationResult result = new SimulationResult(structure, bestIndividual.getAssignment(), bestIndividual.getScore());
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

  // Tuple class for the population lists
  public static class Individual {
    private final Assignment assignment;
    private Double score;

    Individual(Assignment assignment) {
      this.assignment = assignment;
    }

    public Assignment getAssignment() {
      return this.assignment;
    }

    public void setScore(Double score) {
      this.score = score;
    }

    public Double getScore() {
      return score;
    }
  }
}
