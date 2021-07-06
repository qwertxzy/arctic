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

public class GeneticSearch extends AssignmentSearchAlgorithm {

  private static final Logger logger = LoggerFactory.getLogger(GeneticSearch.class);

  // TODO: get these parameters from the map.config
  private static final int populationSize = 1000;
  private static final int eliteNumber = 5;
  private static final int crossoverCount = 500;
  private static final int iterationCount = 100;
  private static final double mutationRate = 0.02;

  public GeneticSearch(Circuit structure, GateLibrary lib, MappingConfiguration mapConfig, SimulationConfiguration simConfig) {
    super(structure, lib, mapConfig, simConfig);
  }

  public SimulationResult assign() {

    // Initialize gate library & simulator

    HashMap<LogicType, List<GateRealization>> realizations = gateLib.getRealizations();
    SimulatorInterface simulator = new SimulatorInterface(simConfig, gateLib.getSourceFile());
    simulator.initSimulation(structure);
    RandomAssigner randomAssigner = new RandomAssigner(gateLib, structure);
    Random random = new Random();

    // Generate initial population

    List<Assignment> currentPopulation = new ArrayList<>();
    List<Assignment> nextPopulation = new ArrayList<>();
    HashMap<Assignment, Double> fitnessLookup = new HashMap<>();
    Assignment bestAssignment = null;

    for (int i = 0; i < populationSize; i++) {
      currentPopulation.add(randomAssigner.getNextAssignment());
    }

    // Loop until exit condition is met

    currentIteration = 0;
    int simCount = 0; // TODO: update simCount accordingly
    while (checkExitCondition()) {
      logger.info("Beginning computation of generation " + currentIteration);

      // Calculate fitness of current population
      // TODO: multithreading the simulation should be an easy way to achieve a significant speed-up per generation

      currentPopulation.forEach(assignment -> {if (!fitnessLookup.containsKey(assignment)) {fitnessLookup.put(assignment, assignment.isValid() ? simulator.simulate(assignment) : 0.0);}});
      currentPopulation.sort((a1, a2) -> fitnessLookup.get(a1).compareTo(fitnessLookup.get(a2)) * (-1)); // TODO: will not sort according to mapConfig

      logger.info("Best score of the current generation: " + fitnessLookup.get(currentPopulation.get(0)));
      logger.info("Average score of the current generation: " + ( currentPopulation.stream().map(fitnessLookup::get).reduce(0.0, Double::sum) / populationSize ));

      // Select the best n as elites to be carried over to the next generation

      for (int i = 0; i < eliteNumber; i++) {
        nextPopulation.add(currentPopulation.get(i));
      }

      // Take n best individuals, generate n / 2 crossed over children
      // TODO: check whether the simulator will score double gate implementation usage with 0

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

      bestAssignment = currentPopulation.get(1);
      Collections.copy(currentPopulation, nextPopulation);
      nextPopulation.clear();
    }

    SimulationResult result = new SimulationResult(structure, bestAssignment, fitnessLookup.get(bestAssignment));
    result.setNeededSimulations(simCount);
    simulator.shutdown();
    return result;
  }

  private int currentIteration;

  private boolean checkExitCondition() {
    currentIteration++;
    // TODO: Switch depending on exit after n iterations or achieved score or whatever
    return currentIteration <= iterationCount;
  }
}
