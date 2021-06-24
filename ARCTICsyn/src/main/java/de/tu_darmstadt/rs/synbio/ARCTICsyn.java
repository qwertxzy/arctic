package de.tu_darmstadt.rs.synbio;

import com.fasterxml.jackson.databind.ObjectMapper;
import de.tu_darmstadt.rs.synbio.common.circuit.Circuit;
import de.tu_darmstadt.rs.synbio.common.library.GateLibrary;
import de.tu_darmstadt.rs.synbio.mapping.MappingConfiguration;
import de.tu_darmstadt.rs.synbio.mapping.search.AssignmentSearchAlgorithm;
import de.tu_darmstadt.rs.synbio.simulation.SimulationConfiguration;
import de.tu_darmstadt.rs.synbio.simulation.SimulationResult;
import de.tu_darmstadt.rs.synbio.synthesis.SynthesisConfiguration;
import de.tu_darmstadt.rs.synbio.synthesis.enumeration.Enumerator;
import de.tu_darmstadt.rs.synbio.synthesis.util.TruthTable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class ARCTICsyn {

    private static final Logger logger = LoggerFactory.getLogger(ARCTICsyn.class);

    private final TruthTable inputTruthTable;
    private final File outputDir;
    private final File gateLibraryFile;

    private final SynthesisConfiguration synConfig;
    private final MappingConfiguration mapConfig;
    private final SimulationConfiguration simConfig;

    public ARCTICsyn(TruthTable inputTruthTable, String libraryPath, SynthesisConfiguration synConfig,
                     MappingConfiguration mapConfig, SimulationConfiguration simConfig) throws IOException {

        this.inputTruthTable = inputTruthTable;
        this.synConfig = synConfig;
        this.mapConfig = mapConfig;
        this.simConfig = simConfig;

        /* output dir handling */

        outputDir = (synConfig.getOutputDir() == null) ? new File("run_" + System.currentTimeMillis() + "_" + inputTruthTable.toString()) :
                new File(synConfig.getOutputDir(), "run_" + System.currentTimeMillis() + "_" + inputTruthTable.toString());

        if (!this.outputDir.mkdirs())
            throw new IOException("Error creating output directory " + outputDir.getAbsolutePath());

        /* library file handling */

        gateLibraryFile = new File(libraryPath);

        if (!gateLibraryFile.exists())
            throw new IOException("Primitive gate library file " + gateLibraryFile + " does not exist.");
    }

    public void synthesize() {

        /* initialize gate library */

        final GateLibrary gateLib = new GateLibrary(gateLibraryFile, new Double[]{0.9,0.1,0.0});
        logger.info("Loaded gate library " + gateLib.getSourceFile() + ".");

        /* circuit enumeration */

        logger.info("Enumeration of circuit variants...");
        Enumerator enumerator = new Enumerator(gateLib, inputTruthTable, synConfig.getMaxDepth(), synConfig.getMaxWeight(), synConfig.getWeightRelaxation());
        enumerator.enumerate();
        List<Circuit> circuits = new ArrayList<>(enumerator.getResultCircuits().values());

        if (circuits.isEmpty()) {
            logger.info("No circuit structures found.");
            return;
        }

        logger.info("Found " + circuits.size() + " circuits.");

        circuits.sort(Circuit::compareTo);

        if (circuits.size() > synConfig.getLimitStructuresNum()) {
            circuits = circuits.subList(circuits.size() - synConfig.getLimitStructuresNum(), circuits.size());
        }

        for (int i = 0; i < circuits.size(); i ++) {
            Circuit circ = circuits.get(i);
            circ.setIdentifier("structure_" + i);
            circ.print(new File(outputDir, inputTruthTable.toString() + "_" + circ.getIdentifier() + ".dot"));
            circ.save(new File(outputDir, inputTruthTable.toString() + "_" + circ.getIdentifier() + ".json"));
        }

        /* technology mapping */

        if (simConfig.isSimEnabled()) {

            logger.info("Technology mapping...");

            SimulationResult bestResult = null;

            for (int i = circuits.size() - 1; i >= 0; i--) {

                AssignmentSearchAlgorithm sim = mapConfig.getSearchAlgorithm(circuits.get(i), gateLib, simConfig);

                try {
                    SimulationResult result = sim.assign();

                    if (bestResult == null || result.getScore() > bestResult.getScore())
                        bestResult = result;

                } catch (Exception e) {
                    e.printStackTrace();
                }
            }

            if (bestResult != null) {
                bestResult.getStructure().save(new File(outputDir, "best_" + bestResult.getStructure().getIdentifier() + ".json"));
                bestResult.getStructure().print(new File(outputDir, "best_" + bestResult.getStructure().getIdentifier() + ".dot"));

                try {
                    ObjectMapper mapper = new ObjectMapper();
                    mapper.writerWithDefaultPrettyPrinter().writeValue(new File(outputDir, "best_" + bestResult.getStructure().getIdentifier() + "_assignment.json"), bestResult.getAssignment());
                } catch (Exception e) {
                    e.printStackTrace();
                }

                logger.info("Finished. Result:");
                logger.info(bestResult.getStructure().getIdentifier() + "," + bestResult.getScore() + "," + bestResult.getStructure().getWeight() + "," + bestResult.getAssignment().getIdentifierMap().toString());
            }
        }
        System.exit(0);
    }
}
