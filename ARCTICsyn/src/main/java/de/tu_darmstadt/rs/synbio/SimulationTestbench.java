package de.tu_darmstadt.rs.synbio;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.node.ObjectNode;
import de.tu_darmstadt.rs.synbio.mapping.MappingConfiguration;
import de.tu_darmstadt.rs.synbio.mapping.search.AssignmentSearchAlgorithm;
import de.tu_darmstadt.rs.synbio.common.circuit.Circuit;
import de.tu_darmstadt.rs.synbio.common.circuit.CircuitDeserializer;
import de.tu_darmstadt.rs.synbio.simulation.SimulationConfiguration;
import de.tu_darmstadt.rs.synbio.common.library.GateLibrary;
import de.tu_darmstadt.rs.synbio.simulation.SimulationResult;
import org.apache.commons.cli.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.PrintWriter;
import java.util.Arrays;

public class SimulationTestbench {

    private static final Logger logger = LoggerFactory.getLogger(SimulationTestbench.class);

    private static final String mappingConfigFile = "map.config";
    private static final String simulationConfigFile = "sim.config";

    private static final int numRepetitions = 1;

    public static void main(String[] args) throws Exception {

        Options options = new Options();

        Option inputDirString = new Option("i", "inputPath", true, "path to the input directory or file");
        options.addOption(inputDirString);
        Option gateLibraryFile = new Option("l", "library", true, "path of the gate library file");
        options.addOption(gateLibraryFile);
        Option proxWeightsOpt = new Option("w", "proxWeights", true, "weights for the gate proximity measure");
        options.addOption(proxWeightsOpt);

        CommandLineParser parser = new DefaultParser();
        HelpFormatter formatter = new HelpFormatter();
        CommandLine cmd;

        try {
            cmd = parser.parse(options, args);
        } catch (ParseException e) {
            logger.error(e.getMessage());
            formatter.printHelp("AssignmentBenchmark", options);
            System.exit(1);
            return;
        }

        if (!cmd.hasOption("inputPath") || !cmd.hasOption("library")) {
            logger.error("Input directory or gate library file not given!");
            formatter.printHelp("enuMap", options);
            System.exit(1);
            return;
        }

        Double[] proxWeights = {1.0, 1.0, 1.0};

        if (cmd.hasOption("proxWeights")) {
            String[] pw = cmd.getOptionValue("proxWeights").split(",");
            proxWeights = Arrays.stream(pw).map(Double::valueOf).toArray(Double[]::new);
        }

        MappingConfiguration mapConfig = new MappingConfiguration(mappingConfigFile);
        SimulationConfiguration simConfig = new SimulationConfiguration(simulationConfigFile);

        GateLibrary gateLib = new GateLibrary(new File(cmd.getOptionValue("library")), proxWeights);

        File inputPath = new File(cmd.getOptionValue("inputPath"));

        File[] directoryListing;

        if (inputPath.isDirectory()) {

            directoryListing = inputPath.listFiles();

            if (directoryListing == null) {
                logger.info("Empty input directory.");
                return;
            }

            Arrays.sort(directoryListing);
        } else {
            directoryListing = new File[1];
            directoryListing[0] = inputPath;
        }

        File output = new File(inputPath.isDirectory() ? inputPath : inputPath.getParentFile(), "results_" + System.currentTimeMillis() + ".txt");
        PrintWriter out;
        try {
            out = new PrintWriter(output);
        } catch (Exception e) {
            e.printStackTrace();
            return;
        }

        long startTime = System.currentTimeMillis();

        for (File child : directoryListing) {

            // test if json
            if (!child.getName().endsWith(".json"))
                continue;

            Circuit structure = null;

            final ObjectNode node;
            ObjectMapper mapper = new ObjectMapper();
            CircuitDeserializer circuitDeserializer = new CircuitDeserializer(Circuit.class);

            try {
                node = mapper.readValue(child, ObjectNode.class);

                if (node.has("graph")) {
                    structure = circuitDeserializer.deserializeString(node.get("graph").toString());
                }

            } catch (Exception e) {
                e.printStackTrace();
            }

            if (structure != null) {

                try {
                    out.print(child.getName());
                } catch (Exception e) {
                    e.printStackTrace();
                }

                int neededSims = 0;

                for (int i = 0; i < numRepetitions; i ++) {
                    AssignmentSearchAlgorithm search = mapConfig.getSearchAlgorithm(structure, gateLib, simConfig);
                    SimulationResult result = search.assign();

                    neededSims += result.getNeededSimulations();

                    logger.info(child.getName() + "," + result.getScore() + "," + result.getStructure().getWeight() +"," + result.getNeededSimulations());

                    //result.getStructure().print(new File(inputPath.isDirectory() ? inputPath : inputPath.getParentFile(),
                    //        "result_" + child.getName() + ".dot"), result.getAssignment());

                    result.getStructure().saveGml(new File(inputPath.isDirectory() ? inputPath : inputPath.getParentFile(),
                            "result_" + child.getName() + ".gml"), result.getAssignment());

                    try {
                        out.print("," + result.getScore() + "," + result.getAssignment().getIdentifierMap());
                        out.flush();
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                }

                try {
                    out.print("," + neededSims + "\n");
                } catch (Exception e) {
                    e.printStackTrace();
                }

            }
        }

        out.println("total time: " + (System.currentTimeMillis() - startTime) + " ms for " + numRepetitions + " repetitions");
        out.close();

        System.exit(0);
    }
}
