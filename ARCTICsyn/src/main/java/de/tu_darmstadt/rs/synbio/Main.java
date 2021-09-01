package de.tu_darmstadt.rs.synbio;

import de.tu_darmstadt.rs.synbio.mapping.MappingConfiguration;
import de.tu_darmstadt.rs.synbio.simulation.SimulationConfiguration;
import de.tu_darmstadt.rs.synbio.synthesis.SynthesisConfiguration;
import de.tu_darmstadt.rs.synbio.synthesis.util.ExpressionParser;
import de.tu_darmstadt.rs.synbio.synthesis.util.TruthTable;
import org.apache.commons.cli.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class Main {

    private static final Logger logger = LoggerFactory.getLogger(Main.class);

    private static final Options options = new Options();
    private static final CommandLineParser parser = new DefaultParser();
    private static final HelpFormatter formatter = new HelpFormatter();

    private static final String synthesisConfigFile = "syn.config";
    private static final String mappingConfigFile = "map.config";
    private static final String simulationConfigFile = "sim.config";

    public static void main(String[] args) throws Exception {

        // parse command line arguments

        /* input */

        Option function = new Option("f", "function", true, "input function");
        options.addOption(function);

        Option truthTable = new Option("t", "truthtable", true, "input truth table");
        options.addOption(truthTable);

        /* library */

        Option gateLibraryFile = new Option("l", "library", true, "path of the gate library file");
        options.addOption(gateLibraryFile);

        CommandLine cmd;

        try {
            cmd = parser.parse(options, args);
        } catch (ParseException e) {
            exit(e.getMessage());
            return;
        }

        // sanity check arguments

        if (!(cmd.hasOption("function")  || cmd.hasOption("truthtable")) || !cmd.hasOption("library")) {
            exit("Input function or gate library file not given!");
        }

        if (cmd.hasOption("function")  && cmd.hasOption("truthtable")) {
            exit("Input function and truth table given!");
        }

        SynthesisConfiguration synConfig = new SynthesisConfiguration(synthesisConfigFile);
        //synConfig.print();
        SimulationConfiguration simConfig = new SimulationConfiguration(simulationConfigFile);
        //simConfig.print();
        MappingConfiguration mapConfig = new MappingConfiguration(mappingConfigFile);
        //mapConfig.print();

        /* input handling */

        TruthTable inputTruthTable;
        System.out.println(cmd.getOptionValue("truthtable"));

        if (cmd.hasOption("function")) {
            inputTruthTable = new TruthTable(ExpressionParser.parse(cmd.getOptionValue("function")));
        } else {
            int ttLength = cmd.getOptionValue("truthtable").length();
            if ((ttLength & (ttLength - 1)) != 0) {
                exit("Length of truth table has to be power of two.");
            }
            String decTT = Integer.parseInt(cmd.getOptionValue("truthtable"), 2) + "," + cmd.getOptionValue("truthtable").length();
            inputTruthTable = new TruthTable(decTT);
        }

        // call main program

        ARCTICsyn syn = new ARCTICsyn(inputTruthTable, cmd.getOptionValue("library"), synConfig, mapConfig, simConfig);
        syn.synthesize();
    }

    private static void exit(String message) {
        logger.error("Error: " + message);
        formatter.printHelp("ARCTICsyn", options);
        System.exit(1);
    }
}
