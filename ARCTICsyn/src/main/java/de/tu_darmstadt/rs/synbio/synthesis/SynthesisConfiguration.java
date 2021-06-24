package de.tu_darmstadt.rs.synbio.synthesis;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import javax.imageio.metadata.IIOMetadataNode;
import java.io.*;
import java.util.Arrays;
import java.util.Properties;

public class SynthesisConfiguration {

    private static final Logger logger = LoggerFactory.getLogger(SynthesisConfiguration.class);

    /* IO fields */

    private String outputDir;

    /* synthesis configuration */

    private int maxDepth;
    private int maxWeight;
    private int weightRelaxation;
    private SynthesisMode synthesisMode;
    private int limitStructuresNum;

    public enum SynthesisMode {
        SUPERGATES, EXHAUSTIVE
    }

    /* supergate generation configuration */

    private File supergateLibrary;
    private boolean exportSupergates;
    private int[] supergateFeasibilities;
    private int maxSupergateDepth;
    private int maxSupergateWeight;

    public SynthesisConfiguration() {}

    public SynthesisConfiguration(String configFile) throws Exception {

        /* config file handling */

        Properties props = new Properties();
        InputStream is = null;
        is = new FileInputStream(configFile);
        props.load(is);


        /* output folder */

        outputDir = props.getProperty("OUTPUT_DIR");

        /* synthesis config handling */

        switch (props.getProperty("SYNTHESIS_MODE")) {
            case "EXHAUSTIVE": synthesisMode = SynthesisMode.EXHAUSTIVE; break;
            case "SUPERGATES": synthesisMode = SynthesisMode.SUPERGATES; break;
            default: throw new IOException("Unknown synthesis mode! (Available modes: " + Arrays.toString(SynthesisMode.values()) + ")");
        }

        maxDepth = Integer.parseInt(props.getProperty("SYNTHESIS_DEPTH"));
        maxWeight = Integer.parseInt(props.getProperty("SYNTHESIS_WEIGHT"));
        weightRelaxation = Integer.parseInt(props.getProperty("SYNTHESIS_WEIGHT_RELAXATION"));
        limitStructuresNum = Integer.parseInt(props.getProperty("SYNTHESIS_LIMIT_STRUCTURES_NUM"));


        /* supergate config handling */

        if (synthesisMode.equals(SynthesisMode.SUPERGATES)) {

            supergateLibrary = new File(props.getProperty("SUPERGATE_LIBRARY"));
            exportSupergates = Boolean.parseBoolean(props.getProperty("SUPERGATE_EXPORT"));

            if (!exportSupergates && !supergateLibrary.exists())
                throw new IOException("Supergate gate library file " + supergateLibrary + " does not exist.");

            if (exportSupergates) {
                String[] feasibilitiyStrings = props.getProperty("SUPERGATE_FEASIBILITIES").split(",");
                supergateFeasibilities = new int[feasibilitiyStrings.length];
                for (int i = 0; i < supergateFeasibilities.length; i++) {
                    supergateFeasibilities[i] = Integer.parseInt(feasibilitiyStrings[i]);
                }
                maxSupergateDepth = Integer.parseInt(props.getProperty("SUPERGATE_DEPTH"));
                maxSupergateWeight = Integer.parseInt(props.getProperty("SUPERGATE_WEIGHT"));
            }

        }
    }

    public void print() {

        logger.info("<-- Configuration -->");

        logger.info("IO:");
        logger.info("\toutput dir.: " + outputDir);

        logger.info("Synthesis:");
        logger.info("\tmode: " + synthesisMode);
        logger.info("\tmax. depth: " + maxDepth);
        logger.info("\tmax. weight: " + maxWeight);
        logger.info("\tweight relaxation: " + weightRelaxation);
        logger.info("\tstructures limit: " + limitStructuresNum);

        if (synthesisMode.equals(SynthesisMode.SUPERGATES)) {
            logger.info("Supergates:");
            logger.info("\tsupergate library: " + supergateLibrary.getAbsolutePath());
            logger.info("\texport library: " + exportSupergates);
            logger.info("\tfeasibilities: " + Arrays.toString(supergateFeasibilities));
            logger.info("\tmax. depth: " + maxSupergateDepth);
            logger.info("\tmax. weight: " + maxSupergateWeight);
        }
    }

    /* getter */

    public File getSupergateLibrary() {
        return supergateLibrary;
    }

    public boolean exportSupergates() {
        return exportSupergates;
    }

    public String getOutputDir() {
        return outputDir;
    }

    public int getMaxDepth() {
        return maxDepth;
    }

    public int getMaxWeight() {
        return maxWeight;
    }

    public int getWeightRelaxation() { return weightRelaxation; }

    public SynthesisMode getSynthesisMode() {
        return synthesisMode;
    }

    public int getLimitStructuresNum() {
        return limitStructuresNum;
    }

    public int[] getSupergateFeasibilities() {
        return supergateFeasibilities;
    }

    public int getMaxSupergateDepth() {
        return maxSupergateDepth;
    }

    public int getMaxSupergateWeight() {
        return maxSupergateWeight;
    }
}
