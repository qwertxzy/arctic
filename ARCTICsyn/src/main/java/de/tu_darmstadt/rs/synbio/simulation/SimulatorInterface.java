package de.tu_darmstadt.rs.synbio.simulation;

import com.fasterxml.jackson.databind.ObjectMapper;
import de.tu_darmstadt.rs.synbio.mapping.Assignment;
import de.tu_darmstadt.rs.synbio.common.circuit.Circuit;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.Map;

public class SimulatorInterface {

    private static final Logger logger = LoggerFactory.getLogger(SimulatorInterface.class);

    private final String pythonBinary;
    private final File simulatorPath;
    private final String simScript;
    private final String simInitArgs;
    private final String simArgs;
    private final File library;

    private Process simProcess;
    private BufferedReader reader;
    private BufferedWriter writer;
    private final ObjectMapper mapper = new ObjectMapper();

    public SimulatorInterface(SimulationConfiguration config, File gateLibrary) {
        pythonBinary = config.getPythonBinary();
        simulatorPath = config.getSimPath();
        simScript = config.getSimScript();
        simInitArgs = config.getSimInitArgs();
        simArgs = config.getSimArgs();
        library = gateLibrary;
    }

    public SimulatorInterface(String pythonBinary, File simPath, String simScript, String simInitArgs, String simArgs, File gateLibrary) {
        this.pythonBinary = pythonBinary;
        simulatorPath = simPath;
        this.simScript = simScript;
        this.simInitArgs = simInitArgs;
        this.simArgs = simArgs;
        this.library = gateLibrary;
    }

    public void initSimulation(Circuit circuit) {

        if (simProcess!= null && simProcess.isAlive())
            simProcess.destroy();

        try {
            String structureFileName = "structure_tid" + Thread.currentThread().getId() + "_" + System.nanoTime() + ".json";
            File structureFile = new File(simulatorPath, structureFileName);
            circuit.save(structureFile);

            ProcessBuilder pb = new ProcessBuilder(pythonBinary, simScript, "s_path=" + structureFileName + " lib_path=" + library.getAbsolutePath() + " " + simInitArgs);
            pb.directory(simulatorPath);
            simProcess = pb.start();

            reader = new BufferedReader(new InputStreamReader(simProcess.getInputStream()));
            writer = new BufferedWriter(new OutputStreamWriter(simProcess.getOutputStream()));

            while (!reader.readLine().startsWith("ready:"));

            structureFile.delete();

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public Double simulate(Assignment assignment) {

        Map<String, String> assignmentIdentifiers = assignment.getIdentifierMap();

        double score = 0.0;

        // TODO restore this (no braces, no termination)
        if (!simProcess.isAlive()) {
            logger.error("sim process aborted");
            return score;
        }

        try {
            String assignmentStr = mapper.writeValueAsString(assignmentIdentifiers);

            writer.write("start " + simArgs + " assignment=" + assignmentStr);
            writer.newLine();
            writer.flush();

            String scoreStr = reader.readLine();
            if (scoreStr.startsWith("O ")) {
                scoreStr = scoreStr.substring(2);
            }

            score = Double.parseDouble(scoreStr);

        } catch (Exception e) {
            e.printStackTrace();
        }

        return score;
    }

    public void shutdown() {
        simProcess.destroy();
    }
}
