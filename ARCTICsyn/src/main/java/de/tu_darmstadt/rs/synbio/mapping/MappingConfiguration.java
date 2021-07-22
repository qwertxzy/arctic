package de.tu_darmstadt.rs.synbio.mapping;

import de.tu_darmstadt.rs.synbio.common.circuit.Circuit;
import de.tu_darmstadt.rs.synbio.common.library.GateLibrary;
import de.tu_darmstadt.rs.synbio.mapping.search.*;
import de.tu_darmstadt.rs.synbio.simulation.SimulationConfiguration;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Arrays;
import java.util.Properties;

public class MappingConfiguration {

    private static final Logger logger = LoggerFactory.getLogger(MappingConfiguration.class);

    private final MappingConfiguration.SearchAlgorithm searchAlgorithm;
    private final MappingConfiguration.OptimizationType optimizationType;
    private final int populationSize;
    private final int eliteNumber;
    private final int crossoverCount;
    private final int iterationCount;
    private final double mutationRate;

    public enum SearchAlgorithm {
        EXHAUSTIVE, TABU, ANNEALING, DAC, GENETIC
    }

    public enum OptimizationType {
        MAXIMIZE {
            @Override
            public boolean compare(double current, double candidate) {
                return candidate > current;
            }
        },

        MINIMIZE {
            @Override
            public boolean compare(double current, double candidate) {
                return candidate < current;
            }
        };

        public abstract boolean compare(double current, double candidate);
    }

    public MappingConfiguration(String configFile) throws Exception {

        /* config file handling */

        Properties props = new Properties();
        InputStream is;
        is = new FileInputStream(configFile);
        props.load(is);

        switch (props.getProperty("SEARCH_ALGORITHM")) {
            case "EXHAUSTIVE":
                searchAlgorithm = SearchAlgorithm.EXHAUSTIVE;
                break;
            case "TABU":
                searchAlgorithm = SearchAlgorithm.TABU;
                break;
            case "ANNEALING":
                searchAlgorithm = SearchAlgorithm.ANNEALING;
                break;
            case "DAC":
                searchAlgorithm = SearchAlgorithm.DAC;
                break;
            case "GENETIC":
                searchAlgorithm = SearchAlgorithm.GENETIC;
                break;
            default:
                throw new IOException("Unknown search algorithm! (Available algorithms: " + Arrays.toString(SearchAlgorithm.values()) + ")");
        }

        switch (props.getProperty("OPTIMIZATION_TYPE")) {
            case "MAXIMIZE":
                optimizationType = OptimizationType.MAXIMIZE;
                break;
            case "MINIMIZE":
                optimizationType = OptimizationType.MINIMIZE;
                break;
            default:
                throw new IOException("Unknown optimization type! (Available types: " + Arrays.toString(OptimizationType.values()) + ")");
        }

        // Parse GENETIC specific parameters
        this.populationSize = Integer.parseInt(props.getProperty("POPULATION_SIZE", "0"));
        this.eliteNumber = Integer.parseInt(props.getProperty("ELITE_NUMBER", "0"));
        this.crossoverCount = Integer.parseInt(props.getProperty("CROSSOVER_COUNT", "0"));
        this.iterationCount = Integer.parseInt(props.getProperty("ITERATION_COUNT", "0"));
        this.mutationRate = Double.parseDouble(props.getProperty("MUTATION_RATE", "0"));
    }

    public void print() {
        logger.info("\tsearch algorithm: " + searchAlgorithm.name());
        logger.info("\toptimization type: " + optimizationType.name());
    }

    public OptimizationType getOptimizationType() {
        return optimizationType;
    }

    public int getPopulationSize() {
        return populationSize;
    }

    public int getEliteNumber() {
        return eliteNumber;
    }

    public int getCrossoverCount() {
        return crossoverCount;
    }

    public int getIterationCount() {
        return iterationCount;
    }

    public double getMutationRate() {
        return mutationRate;
    }

    /* factories */

    public AssignmentSearchAlgorithm getSearchAlgorithm(Circuit structure, GateLibrary lib, SimulationConfiguration simConfig) {

        switch (searchAlgorithm) {
            case ANNEALING: return new SimulatedAnnealingSearch(structure, lib, this, simConfig);
            case GENETIC: return new GeneticSearch(structure, lib, this, simConfig);
            default: return new ExhaustiveSearch(structure, lib, this, simConfig);
        }
    }
}
