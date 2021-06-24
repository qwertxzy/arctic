package de.tu_darmstadt.rs.synbio.simulation;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Properties;

public class SimulationConfiguration {

    private static final Logger logger = LoggerFactory.getLogger(SimulationConfiguration.class);

    /* simulation configuration */

    private final boolean simEnabled;
    private String pythonBinary;
    private boolean simLimitThreads;
    private int simLimitThreadsNum;
    private File simPath;
    private String simScript;
    private String simInitArgs;
    private String simArgs;

    public SimulationConfiguration(String configFile) throws Exception {

        /* config file handling */

        Properties props = new Properties();
        InputStream is = null;
        is = new FileInputStream(configFile);
        props.load(is);

        /* simulation config handling */

        simEnabled = Boolean.parseBoolean(props.getProperty("SIM_ENABLED"));

        if (simEnabled) {

            pythonBinary = props.getProperty("PYTHON_BINARY");

            simLimitThreads = Boolean.parseBoolean(props.getProperty("SIM_LIMIT_THREADS"));

            if (simLimitThreads)
                simLimitThreadsNum = Integer.parseInt(props.getProperty("SIM_LIMIT_THREADS_NUM"));

            simPath = new File(props.getProperty("SIM_PATH"));

            simScript = props.getProperty("SIM_SCRIPT");

            if (!new File(simPath, simScript).exists())
                throw new IOException("Simulator script " + simScript + " does not exist.");

            simInitArgs = props.getProperty("SIM_INIT_ARGS");

            simArgs = props.getProperty("SIM_ARGS");
        }
    }

    public void print() {

        if (simEnabled) {
            logger.info("Simulation:");
            logger.info("\tpython binary: " + pythonBinary);
            logger.info("\tsimulator path: " + simPath);
            logger.info("\tsimulator script: " + simScript);
            logger.info("\tsimulator initialization arguments: " + simInitArgs);
            logger.info("\tsimulator arguments: " + simArgs);
            logger.info("\tlimit sim. threads: " + simLimitThreads);
            if (simLimitThreads)
                logger.info("\tthread limit: " + simLimitThreadsNum);
        }
    }

    /* getter */

    public boolean isSimEnabled() {
        return simEnabled;
    }

    public String getPythonBinary() {
        return pythonBinary;
    }

    public boolean simLimitThreads() {
        return simLimitThreads;
    }

    public int getSimLimitThreadsNum() {
        return simLimitThreadsNum;
    }

    public File getSimPath() {
        return simPath;
    }

    public String getSimScript() {
        return simScript;
    }

    public String getSimInitArgs() { return simInitArgs; }

    public String getSimArgs() {
        return simArgs;
    }
}
