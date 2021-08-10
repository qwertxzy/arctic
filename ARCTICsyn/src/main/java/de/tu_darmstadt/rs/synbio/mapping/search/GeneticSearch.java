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

import java.io.PrintWriter;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicLong;
import java.util.stream.Collectors;

public class GeneticSearch extends AssignmentSearchAlgorithm {

  private static final Logger logger = LoggerFactory.getLogger(GeneticSearch.class);

  private final int populationSize;
  private final int eliteNumber ;
  private final int crossoverCount;
  private final int iterationCount;
  private final double mutationRate;
  private final Map<LogicType, List<String>> geneEncoding; // TODO: change String to BitSet to improve performance
  private final Map<LogicType, List<GateRealization>> realizations;


  public GeneticSearch(Circuit structure, GateLibrary lib, MappingConfiguration mapConfig, SimulationConfiguration simConfig) {
    super(structure, lib, mapConfig, simConfig);

    this.populationSize = mapConfig.getPopulationSize();
    this.eliteNumber = mapConfig.getEliteNumber();
    this.crossoverCount = mapConfig.getCrossoverCount();
    this.iterationCount = mapConfig.getIterationCount();
    this.mutationRate = mapConfig.getMutationRate();

    // Generate gene encoding lookup
    this.realizations = gateLib.getRealizations();
    this.geneEncoding = new HashMap<>();

    for (LogicType type : realizations.keySet()) {
      List<String> binaryRepresentations = new ArrayList<>();
      int realizationAmount = realizations.get(type).size();
      int bitWidth = (int) Math.ceil(Math.log(realizationAmount / Math.log(2))) + 1;

      List<Integer> grayCodes = grayCode(bitWidth);
      for (int i = 0; i < realizationAmount; i++) {
        String binaryString = Integer.toBinaryString(grayCodes.get(i));
        String paddedString = String.format("%" + bitWidth + "s", binaryString).replaceAll(" ", "0");
        binaryRepresentations.add(paddedString);
      }

      geneEncoding.put(type, binaryRepresentations);
    }
  }

  public SimulationResult assign() {

    // Initialize gate library & simulators

    StringBuilder detailCSV = new StringBuilder(); // CSV export of all fitness data for further analysis

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

    List<GeneticSearchIndividual> currentPopulation = new ArrayList<>();
    List<GeneticSearchIndividual> nextPopulation = new ArrayList<>();

    // Generate initial population

    for (int i = 0; i < populationSize; i++) {
      GeneticSearchIndividual individual = new GeneticSearchIndividual(randomAssigner.getNextAssignment());
      individual.setEncodedAssignment(GeneticSearchIndividual.geneticEncode(realizations, geneEncoding, individual.getAssignment()));
      currentPopulation.add(individual);
    }
    GeneticSearchIndividual bestIndividual = currentPopulation.get(0);

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
        List<GeneticSearchIndividual> slice = currentPopulation.subList((int) Math.ceil(i * sliceLength), (int) Math.ceil((i + 1) * sliceLength));
        workers.add(new GeneticSearchWorker(simulators.get(i), slice, realizations, geneEncoding, simCount, invalidCount));
      }

      ExecutorService executor = Executors.newFixedThreadPool(availableProcessors);
      // TODO: these don't need to be reconstructed each loop

      try {
        executor.invokeAll(workers);
      } catch (InterruptedException e) {
        logger.error(e.getMessage());
      }

      // Extract the best individual
      // TODO: make this respect the OptimizationType here & above in the parent selection
      GeneticSearchIndividual generationBestIndividual = currentPopulation.stream().max(Comparator.comparing(GeneticSearchIndividual::getScore)).get();
      bestIndividual = (generationBestIndividual.getScore() > bestIndividual.getScore() ? generationBestIndividual : bestIndividual);

      logger.info(
              currentIteration +
                      "," + invalidCount.get() +
                      "," + bestIndividual.getScore() +
                      "," + ( (currentPopulation.subList(0, 5).stream().map(GeneticSearchIndividual::getScore).reduce(0.0, Double::sum)) / 5.0 ) +
                      "," + ( currentPopulation.stream().map(GeneticSearchIndividual::getScore).reduce(0.0, Double::sum) / populationSize )
      );

      detailCSV.append(currentPopulation.stream().map(GeneticSearchIndividual::getScore).map(Objects::toString).collect(Collectors.joining(","))).append("\n");

      // Select the best n as elites to be carried over to the next generation
      // TODO: nonsense, list is not sorted anymore
      for (int i = 0; i < eliteNumber; i++) {
        nextPopulation.add(currentPopulation.get(i));
      }

      // Apply stochastic universal sampling to choose parents via roulette wheel selection

      double totalFitness = currentPopulation.stream().map(GeneticSearchIndividual::getScore).reduce(0.0, Double::sum);
      double intervalDistance = totalFitness / crossoverCount;
      double startPoint = intervalDistance * random.nextDouble();

      List<Double> pointers = new ArrayList<>();
      for (int i = 0; i < crossoverCount; i++) {
        pointers.add(startPoint + i * intervalDistance);
      }

      List<GeneticSearchIndividual> parents = new ArrayList<>();
      for (Double point : pointers) {
        int i = 0;
        while (currentPopulation.subList(0, i + 1).stream().map(GeneticSearchIndividual::getScore).reduce(0.0, Double::sum) < point) { // TODO: maybe scale the fitness, fit solutions dominate the pool too much
          i++;
        }
        parents.add(currentPopulation.get(i));
      }

      // Cross parents with random crossover point

      for (int i = 0; i < parents.size(); i += 2) {
        GeneticSearchIndividual firstParent = parents.get(i);
        GeneticSearchIndividual secondParent = parents.get(i + 1);

        StringBuilder encodedFirstChild = new StringBuilder();
        StringBuilder encodedSecondChild = new StringBuilder();

        int crossoverPoint = random.nextInt(firstParent.getEncodedAssignment().length());

        for (int j = 0; j < firstParent.getEncodedAssignment().length(); j++) {
          if (crossoverPoint < j) {
            encodedFirstChild.append(firstParent.getEncodedAssignment().charAt(j));
            encodedSecondChild.append(secondParent.getEncodedAssignment().charAt(j));
          } else {
            encodedFirstChild.append(secondParent.getEncodedAssignment().charAt(j));
            encodedSecondChild.append(firstParent.getEncodedAssignment().charAt(j));
          }
        }

        GeneticSearchIndividual firstChild = new GeneticSearchIndividual(firstParent.getAssignment()); // Will not match encodedAssignment, but is needed for the keyset
        firstChild.setEncodedAssignment(encodedFirstChild.toString());
        nextPopulation.add(firstChild);

        GeneticSearchIndividual secondChild = new GeneticSearchIndividual(secondParent.getAssignment());
        secondChild.setEncodedAssignment(encodedSecondChild.toString());
        nextPopulation.add(secondChild); // chaansu
      }

      // Apply random mutation (select a random equivalent gate implementation for random individuals)

      for (GeneticSearchIndividual individual : nextPopulation) {
        if (random.nextDouble() <= mutationRate) {
          StringBuilder mutatingGenome = new StringBuilder(individual.getEncodedAssignment());

          int mutatingIndex = random.nextInt(mutatingGenome.length());
          char mutatingChar = mutatingGenome.charAt(mutatingIndex);

          mutatingGenome.setCharAt(mutatingIndex, (mutatingChar == '0' ? '1' : '0'));
        }
      }

      // Fill up the rest with new random individuals

      while (nextPopulation.size() < populationSize) {
        GeneticSearchIndividual individual = new GeneticSearchIndividual(randomAssigner.getNextAssignment());
        individual.setEncodedAssignment(GeneticSearchIndividual.geneticEncode(realizations, geneEncoding, individual.getAssignment()));
        nextPopulation.add(individual);
      }

      // Move the next generation's individuals to the current generation
      currentPopulation = nextPopulation;
      nextPopulation = new ArrayList<>();
    }

    SimulationResult result = new SimulationResult(structure, bestIndividual.getAssignment(), bestIndividual.getScore());
    result.setNeededSimulations(simCount.get());

    for (SimulatorInterface sim : simulators) {
      sim.shutdown();
    }

    // Export the generated CSV output

    try {
      PrintWriter out = new PrintWriter("details.csv");
      out.println(detailCSV);
    } catch (Exception e) {
      logger.error(e.getMessage());
    }

    return result;
  }

  private int currentIteration;

  private boolean checkExitCondition() {
    currentIteration++;
    // TODO: Switch depending on exit after n iterations or achieved score or whatever
    return currentIteration <= iterationCount;
  }

  // Taken from https://www.programcreek.com/2014/05/leetcode-gray-code-java/
  private List<Integer> grayCode(int n) {
    if(n == 0) {
      List<Integer> result = new ArrayList<>();
      result.add(0);
      return result;
    }

    List<Integer> result = grayCode(n - 1);
    int numToAdd = 1 << (n - 1);

    for(int i = result.size() - 1; i >= 0; i--){ //iterate from last to first
      result.add(numToAdd + result.get(i));
    }

    return result;
  }
}
