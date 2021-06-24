package de.tu_darmstadt.rs.synbio.mapping.search;

import de.tu_darmstadt.rs.synbio.mapping.MappingConfiguration;
import de.tu_darmstadt.rs.synbio.mapping.assigner.ExhaustiveAssigner;
import de.tu_darmstadt.rs.synbio.simulation.SimulationConfiguration;
import de.tu_darmstadt.rs.synbio.common.circuit.Circuit;
import de.tu_darmstadt.rs.synbio.common.library.GateLibrary;
import de.tu_darmstadt.rs.synbio.simulation.SimulationResult;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.PrintWriter;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class ExhaustiveSearch extends AssignmentSearchAlgorithm {

    private static final Logger logger = LoggerFactory.getLogger(ExhaustiveSearch.class);

    private final ExhaustiveAssigner assigner;
    private final ExhaustiveLogger exhaustiveLogger;

    public ExhaustiveSearch(Circuit structure, GateLibrary lib, MappingConfiguration mapConfig, SimulationConfiguration simConfig) {
        super(structure, lib, mapConfig, simConfig);
        assigner = new ExhaustiveAssigner(lib, structure);
        exhaustiveLogger = new ExhaustiveLogger();
    }

    public SimulationResult assign() {

        List<ExhaustiveSearchWorker> workers = new ArrayList<>();
        int maxThreads = Runtime.getRuntime().availableProcessors() - 1;
        int availableProcessors = simConfig.simLimitThreads() ? Math.min(simConfig.getSimLimitThreadsNum(), maxThreads) : maxThreads;

        logger.info("Simulating \"" + structure.getIdentifier() + "\" (up to " + assigner.getNumTotalPermutations() + " assignments) with " + availableProcessors + " threads");

        for (int i = 0; i < availableProcessors; i ++) {
            workers.add(new ExhaustiveSearchWorker(exhaustiveLogger, assigner, structure, mapConfig, simConfig, gateLib));
        }

        ExecutorService executor = Executors.newFixedThreadPool(availableProcessors);

        List<Future<SimulationResult>> simResults = Collections.emptyList();

        try {
            simResults = executor.invokeAll(workers);
        } catch (InterruptedException e) {
            logger.error(e.getMessage());
        }

        SimulationResult bestRes = null;

        for (Future<SimulationResult> result : simResults) {

            try {
                SimulationResult res = result.get();

                if (res != null && (bestRes == null || (mapConfig.getOptimizationType().compare(bestRes.getScore(), res.getScore())))) {
                    bestRes = res;
                }

            } catch (Exception e) {
                logger.error(e.getMessage());
            }
        }

        exhaustiveLogger.write(new File("exhaustiveLog_" + System.currentTimeMillis() + ".txt"));

        logger.info("Finished simulating " + structure.getIdentifier() + ", score: " + (bestRes != null ? bestRes.getScore() : 0));

        return bestRes;
    }

    static class ExhaustiveLogger {

        private final List<String> log = new ArrayList<>();
        private File logFile;

        public ExhaustiveLogger() {}

        public synchronized void append(String entry) {
            log.add(entry);
        }

        public void write(File logFile) {

            PrintWriter out;

            try {
                out = new PrintWriter(logFile);

                for (String entry : log) {
                    out.println(entry);
                }
                out.close();
            } catch (Exception e) {
                e.printStackTrace();
            }
        }
    }
}
