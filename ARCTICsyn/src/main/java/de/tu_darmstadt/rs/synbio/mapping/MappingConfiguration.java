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

    public enum SearchAlgorithm {
        EXHAUSTIVE, TABU, ANNEALING, DAC
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
    }

    public void print() {
        logger.info("\tsearch algorithm: " + searchAlgorithm.name());
        logger.info("\toptimization type: " + optimizationType.name());
    }

    public OptimizationType getOptimizationType() {
        return optimizationType;
    }

    /* factories */

    public AssignmentSearchAlgorithm getSearchAlgorithm(Circuit structure, GateLibrary lib, SimulationConfiguration simConfig) {

        switch (searchAlgorithm) {
            case ANNEALING: return new SimulatedAnnealingSearch(structure, lib, this, simConfig);
            default: return new ExhaustiveSearch(structure, lib, this, simConfig);
        }
    }
}
