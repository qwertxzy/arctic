package de.tu_darmstadt.rs.synbio.mapping.search;

import de.tu_darmstadt.rs.synbio.common.LogicType;
import de.tu_darmstadt.rs.synbio.common.circuit.Circuit;
import de.tu_darmstadt.rs.synbio.common.circuit.LogicGate;
import de.tu_darmstadt.rs.synbio.common.library.GateLibrary;
import de.tu_darmstadt.rs.synbio.common.library.GateRealization;
import de.tu_darmstadt.rs.synbio.mapping.Assignment;
import de.tu_darmstadt.rs.synbio.mapping.MappingConfiguration;
import de.tu_darmstadt.rs.synbio.mapping.assigner.RandomAssigner;
import de.tu_darmstadt.rs.synbio.mapping.util.BitField;
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
  private final int eliteNumber;
  private final int crossoverCount;
  private final double mutationRate;

  private final int iterationCount;
  private final double minimumVariety;

  private int consecInvalidThresh;
  private int consecStaleThresh;

  private final Map<LogicType, List<BitField>> geneEncoding;
  private final Map<LogicType, List<GateRealization>> realizations;

  public GeneticSearch(Circuit structure, GateLibrary lib,
                       MappingConfiguration mapConfig, SimulationConfiguration simConfig) {

    super(structure, lib, mapConfig, simConfig);

    // Set genetic parameters based on circuit size
    this.populationSize = structure.getNumberLogicGates() * 180;
    this.eliteNumber = 1;
    this.crossoverCount = structure.getNumberLogicGates() * 150;
    this.mutationRate = 0.02;

    logger.info("" + populationSize);

    //this.iterationCount = 150;
    this.iterationCount = mapConfig.getIterationCount();
    this.minimumVariety = mapConfig.getMinimumVariety();
    this.consecInvalidThresh = 0;
    this.consecStaleThresh = 0;

    // Generate gene encoding lookup
    this.realizations = gateLib.getRealizations();
    this.geneEncoding = new HashMap<>();

    for (LogicType type : realizations.keySet()) {
      int realizationAmount = realizations.get(type).size();
      int bitWidth = (int) Math.ceil(Math.log(realizationAmount / Math.log(2)) + 1);

      List<BitField> binaryRepresentations = new ArrayList<>();

      List<Integer> grayCodes = grayCode(bitWidth);
      for (int i = 1; i <= realizationAmount; i++) {
        Integer grayNumber = grayCodes.get(i);
        BitField binaryRepresentation = new BitField(bitWidth);
        binaryRepresentation.or(BitField.parseInt(grayNumber));

        binaryRepresentations.add(binaryRepresentation);
      }

      geneEncoding.put(type, binaryRepresentations);
    }
  }

  private long evaluateAndSortPopulation(List<GeneticSearchIndividual> population, List<SimulatorInterface> simulators, Set<LogicGate> gates, ExecutorService executor, AtomicLong simCtr) {
    List<GeneticSearchWorker> workers = new ArrayList<>();
    int workerCount = simulators.size();
    double sliceLength = population.size() / (double) workerCount;
    AtomicLong invalidCount = new AtomicLong(0);

    // Split up current population into even slices to simulate
    for (int i = 0; i < workerCount; i++) {
      List<GeneticSearchIndividual> slice = population.subList((int) Math.ceil(i * sliceLength), (int) Math.ceil((i + 1) * sliceLength));
      workers.add(new GeneticSearchWorker(simulators.get(i), slice, realizations, geneEncoding, gates, simCtr, invalidCount));
    }

    try {
      executor.invokeAll(workers);
    } catch (InterruptedException e) {
      logger.error(e.getMessage());
    }

    // Sort the current population by fitness
    if (mapConfig.getOptimizationType() == MappingConfiguration.OptimizationType.MAXIMIZE) {
      Collections.sort(population);
    } else {
      Collections.sort(population, Collections.reverseOrder());
    }

    return invalidCount.get();
  }

  public SimulationResult assign() {

    // Initialize gate library & simulators

    StringBuilder detailCSV = new StringBuilder(); // CSV export of all fitness data for further analysis

    int maxThreads = Runtime.getRuntime().availableProcessors() - 1;
    int availableProcessors = simConfig.simLimitThreads() ?
        Math.min(simConfig.getSimLimitThreadsNum(), maxThreads) : maxThreads;
    double sliceLength = populationSize / (double) availableProcessors;


    List<SimulatorInterface> simulators = new ArrayList<>();
    for (int i = 0; i < availableProcessors; i++) {
      SimulatorInterface sim = new SimulatorInterface(simConfig, gateLib.getSourceFile());
      sim.initSimulation(structure);
      simulators.add(sim);
    }
    ExecutorService executor = Executors.newFixedThreadPool(availableProcessors);

    RandomAssigner randomAssigner = new RandomAssigner(gateLib, structure);
    Set<LogicGate> assignmentGates = randomAssigner.getNextAssignment().keySet();
    Random random = new Random();

    // Declare population related variables

    List<GeneticSearchIndividual> currentPopulation = new ArrayList<>();
    List<GeneticSearchIndividual> nextPopulation = new ArrayList<>();

    // Generate initial population

    for (int i = 0; i < populationSize; i++) {
      BitField encodedAssignment = GeneticSearchIndividual.geneticEncode(
          realizations, geneEncoding, randomAssigner.getNextAssignment());
      GeneticSearchIndividual individual = new GeneticSearchIndividual(encodedAssignment);
      nextPopulation.add(individual);
    }

    // Atomic long to increment it from simulation threads
    AtomicLong simCount = new AtomicLong(0);

    logger.info("Generation, Invalid, Top, Top5, Average");

    // Evaluate initial population
    long invalidCount = evaluateAndSortPopulation(nextPopulation, simulators, assignmentGates, executor, simCount);

    // Set the best individual
    GeneticSearchIndividual bestIndividual = nextPopulation.get(0);

    // Log stats of the first population
    double averageScore = ( nextPopulation.stream().map(GeneticSearchIndividual::getScore).reduce(0.0, Double::sum) / populationSize );
    logger.info(
            currentIteration +
                    "," + invalidCount +
                    "," + bestIndividual.getScore() +
                    "," + ( (nextPopulation.subList(0, 5).stream().map(GeneticSearchIndividual::getScore).reduce(0.0, Double::sum)) / 5.0 ) +
                    "," + averageScore
    );
    detailCSV.append(nextPopulation.stream().map(GeneticSearchIndividual::getScore)
            .map(Objects::toString).collect(Collectors.joining(","))).append("\n");

    // TODO: restructure the loop to allow consideration of nextPopulation-scores in exit condition
    while (checkExitCondition(invalidCount, nextPopulation, averageScore, bestIndividual.getScore(), currentPopulation)) {
      // Move the next generation's individuals to the current generation
      currentPopulation = nextPopulation;
      nextPopulation = new ArrayList<>();

      // Select the best n as elites to be carried over to the next generation

      if (eliteNumber > 0) {
        nextPopulation.addAll(currentPopulation.subList(0, eliteNumber));
      }

      // Apply rank-based roulette wheel selection

      List<Integer> rankWeights = new ArrayList<>();
      for (int i = 0; i < populationSize; i++) {
        rankWeights.add(calculateRankWeight(i));
      }

      int intervalDistance = rankWeights.stream().reduce(0, Integer::sum) / crossoverCount;
      int startPoint = random.nextInt(intervalDistance);

      List<Integer> pointers = new ArrayList<>();
      for (int i = 0; i < crossoverCount; i++) {
        pointers.add(startPoint + i * intervalDistance);
      }

      List<GeneticSearchIndividual> parents = new ArrayList<>();
      for (Integer point : pointers) {
        int i = 0;
        while (rankWeights.subList(0, i + 1).stream().reduce(0, Integer::sum) < point) {
          i++;
        }
        parents.add(currentPopulation.get(i));
      }

      // Cross parents with random crossover point

      for (int i = 0; i < parents.size() / 2; i++) {
        GeneticSearchIndividual firstParent = parents.get(i);
        GeneticSearchIndividual secondParent = parents.get(parents.size() - i - 1);

        int bitSetWidth = firstParent.getEncodedAssignment().length();

        BitField encodedFirstChild = new BitField(bitSetWidth);
        BitField encodedSecondChild = new BitField(bitSetWidth);

        int crossoverPoint = random.nextInt(firstParent.getEncodedAssignment().length());

        for (int j = 0; j < firstParent.getEncodedAssignment().length(); j++) {
          if (crossoverPoint < j) {
            encodedFirstChild.setBit(j, firstParent.getEncodedAssignment().getBit(j));
            encodedSecondChild.setBit(j, secondParent.getEncodedAssignment().getBit(j));
          } else {
            encodedFirstChild.setBit(j, secondParent.getEncodedAssignment().getBit(j));
            encodedSecondChild.setBit(j, firstParent.getEncodedAssignment().getBit(j));
          }
        }

        GeneticSearchIndividual firstChild = new GeneticSearchIndividual(encodedFirstChild);
        nextPopulation.add(firstChild);

        GeneticSearchIndividual secondChild = new GeneticSearchIndividual(encodedSecondChild);
        nextPopulation.add(secondChild);
      }

      // Apply random mutation

      for (GeneticSearchIndividual individual : nextPopulation) {
        if (random.nextDouble() <= mutationRate) {
          BitField mutatingGenome = individual.getEncodedAssignment();

          int mutatingIndex = random.nextInt(mutatingGenome.length());

          mutatingGenome.flipBit(mutatingIndex);

          individual.setEncodedAssignment(mutatingGenome);
        }
      }

      // Fill up the rest with new random individuals

      while (nextPopulation.size() < populationSize) {
        BitField encodedAssignment = GeneticSearchIndividual.geneticEncode(
            realizations, geneEncoding, randomAssigner.getNextAssignment());
        GeneticSearchIndividual individual = new GeneticSearchIndividual(encodedAssignment);
        nextPopulation.add(individual);
      }

      // Evaluate the resulting population
      invalidCount = evaluateAndSortPopulation(nextPopulation, simulators, assignmentGates, executor, simCount);

      // Extract the best individual
      GeneticSearchIndividual generationBestIndividual = nextPopulation.get(0);
      // TODO: If config allows for minimizing the score, shouldn't this be considered here?
      bestIndividual = (generationBestIndividual.getScore() > bestIndividual.getScore() ? generationBestIndividual : bestIndividual);

      // Log stats of the resulting population
      averageScore = ( nextPopulation.stream().map(GeneticSearchIndividual::getScore).reduce(0.0, Double::sum) / populationSize );
      logger.info(
              currentIteration +
                      "," + invalidCount +
                      "," + bestIndividual.getScore() +
                      "," + ( (nextPopulation.subList(0, 5).stream().map(GeneticSearchIndividual::getScore).reduce(0.0, Double::sum)) / 5.0 ) +
                      "," + averageScore
      );
      detailCSV.append(nextPopulation.stream().map(GeneticSearchIndividual::getScore)
              .map(Objects::toString).collect(Collectors.joining(","))).append("\n");
    }
    currentPopulation = nextPopulation;

      Assignment bestAssignment = GeneticSearchIndividual.geneticDecode(
              realizations, geneEncoding, bestIndividual.getEncodedAssignment(), assignmentGates);
      SimulationResult result = new SimulationResult(structure, bestAssignment, bestIndividual.getScore());
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

  // Function for calculating the roulette wheel weight for a given rank within the population
  private int calculateRankWeight(int rank) {
    // Rank n gets a weight of (popSize - n)^1.9
    return (int) Math.pow((populationSize - rank), 1.9);
  }

  private int currentIteration;

  // TODO set thresholds, evaluate which of the criteria make sense
  private boolean checkExitCondition(long invalidCount, List<GeneticSearchIndividual> population, double averageScore, double topScore, List<GeneticSearchIndividual> lastPopulation) {
    // Criteria #1: Iteration count
    ++currentIteration;
    if (iterationCount > 0 && currentIteration > iterationCount) {
      logger.info("Iteration limit reached - Terminating.");
      return false;
    }
    logger.info("---------------------------------");
    logger.info("Iteration " + currentIteration);

    // Criteria #2: Genetic Variety
    if (invalidCount < 0.02 * populationSize) {
      if (++consecInvalidThresh > 5) {
        logger.info("Invalid count has stagnated - Terminating.");
        return false;
      }
    } else {
      consecInvalidThresh = 0;
    }

    // Criteria #3: Genetic Variety
    if (minimumVariety > 0 && (topScore / averageScore - 1.0) > minimumVariety) {
      logger.info("Population fitness has balanced out - Terminating.");
      return false;
    }
    logger.info("Genetic Variety: " + (topScore / averageScore - 1.0));

    // Criteria #4: No significant change across iterations
    if (lastPopulation == null || lastPopulation.size() == 0)
      return true;
    double lastAverage = ( lastPopulation.stream().map(GeneticSearchIndividual::getScore).reduce(0.0, Double::sum) / populationSize );
    double lastTop = lastPopulation.get(0).getScore();
    double currentTop = population.get(0).getScore();

    if (Math.abs(averageScore / lastAverage - 1.0) < 0.001 && currentTop != lastTop) {
      if (++consecStaleThresh > 5) {
        logger.info("No significant change over several iterations - Terminating.");
        return false;
      }
    }
    else {
      consecStaleThresh = 0;
    }

    // No Exit Criteria were met. Continue
    return true;
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
