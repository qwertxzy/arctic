package de.tu_darmstadt.rs.synbio.common.library;

import com.fasterxml.jackson.databind.ObjectMapper;
import de.tu_darmstadt.rs.synbio.common.LogicType;
import de.tu_darmstadt.rs.synbio.common.circuit.Circuit;
import de.tu_darmstadt.rs.synbio.synthesis.util.TruthTable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.*;

public class GateLibrary {

    private static final Logger logger = LoggerFactory.getLogger(GateLibrary.class);

    private final File sourceFile;

    private final HashMap<TruthTable, Circuit> circuitLibrary = new HashMap<>();
    private final HashMap<LogicType, List<GateRealization>> gateRealizations = new HashMap<>();

    private final Double[] proxNormalization;
    private final Double[] proxWeights;

    public GateLibrary(File libraryFile) {

        this.sourceFile = libraryFile;
        this.proxNormalization = new Double[]{1.0, 1.0, 1.0};
        this.proxWeights = new Double[]{1.0, 1.0, 1.0};

        loadPrimitiveLibrary(libraryFile);
    }

    public GateLibrary(File libraryFile, Double[] proxWeights) {

        this.sourceFile = libraryFile;
        this.proxWeights = proxWeights;

        loadPrimitiveLibrary(libraryFile);

        proxNormalization = calcProxNormalization();
    }

    /* library file handling */

    private void loadPrimitiveLibrary(File libraryFile) {

        HashMap<String, Object>[] parsedRealizations;

        ObjectMapper mapper = new ObjectMapper();

        try {
            parsedRealizations = mapper.readValue(libraryFile, HashMap[].class);
        } catch (Exception e) {
            e.printStackTrace();
            return;
        }

        for (HashMap<String, Object> realization : parsedRealizations) {

            String primitiveIdentifier = (String) Optional.ofNullable(realization.get("primitiveIdentifier"))
                    .orElseThrow(() -> new RuntimeException("Invalid primitive gate library: Key \"primitiveIdentifier\" not found!"));

            String identifier = (String) Optional.ofNullable(realization.get("identifier"))
                    .orElseThrow(() -> new RuntimeException("Invalid primitive gate library: Key \"identifier\" not found!"));

            String altIdentifier = (String) Optional.ofNullable(realization.get("alternative_identifier"))
                    .orElse("");

            String group = (String) Optional.ofNullable(realization.get("group"))
                    .orElseThrow(() -> new RuntimeException("Invalid primitive gate library: Key \"group\" not found!"));

            LinkedHashMap biorep = (LinkedHashMap) Optional.ofNullable(realization.get("biorep"))
                    .orElseThrow(() -> new RuntimeException("Invalid primitive gate library: Key \"biorep\" not found!"));

            LinkedHashMap responseFunction = (LinkedHashMap) Optional.ofNullable(biorep.get("response_function"))
                    .orElseThrow(() -> new RuntimeException("Invalid primitive gate library: Key \"response_function\" not found!"));

            LinkedHashMap parameters = (LinkedHashMap) Optional.ofNullable(responseFunction.get("parameters"))
                    .orElseThrow(() -> new RuntimeException("Invalid primitive gate library: Key \"parameters\" not found!"));

            Optional ymaxOpt = Optional.ofNullable(parameters.get("ymax"));
            Optional yminOpt = Optional.ofNullable(parameters.get("ymin"));
            Optional kOpt = Optional.ofNullable(parameters.get("K"));
            Optional nOpt = Optional.ofNullable(parameters.get("n"));

            GateRealization newRealization;

            if (ymaxOpt.isPresent() && yminOpt.isPresent() && kOpt.isPresent() && nOpt.isPresent()) {

                double ymax = (double) ymaxOpt.get();
                double ymin = (double) yminOpt.get();
                double k = (double) kOpt.get();
                double n = (double) nOpt.get();

                newRealization = new GateRealization(identifier, LogicType.valueOf(primitiveIdentifier), group, altIdentifier,
                        new GateRealization.GateCharacterization(ymax, ymin, k ,n));

            } else {
                newRealization = new GateRealization(identifier, LogicType.valueOf(primitiveIdentifier), group, altIdentifier);
            }

            addToLibrary(newRealization);
        }
    }

    private void addToLibrary(GateRealization element) {

        if (gateRealizations.containsKey(element.getLogicType())) {
            gateRealizations.get(element.getLogicType()).add(element);
        } else {
            gateRealizations.put(element.getLogicType(), new ArrayList<>());
            gateRealizations.get(element.getLogicType()).add(element);

            circuitLibrary.put(new TruthTable(element.getLogicType().getExpression()), new Circuit(element, element.getLogicType().name()));
        }
    }

    private Double[] calcProxNormalization() {

        Double[] normalizationValues = new Double[]{1.0, 1.0, 1.0};

        double ym_max = Double.NEGATIVE_INFINITY;
        double xm_max = Double.NEGATIVE_INFINITY;
        double grad_max = Double.NEGATIVE_INFINITY;
        double ym_min = Double.POSITIVE_INFINITY;
        double xm_min = Double.POSITIVE_INFINITY;
        double grad_min = Double.POSITIVE_INFINITY;

        for (List<GateRealization> allRealizations : gateRealizations.values()) {

            if (allRealizations.size() < 2)
                continue;

            for (GateRealization r1 : allRealizations) {

                ym_max = Math.max(ym_max, r1.getCharacterization().getYm());
                xm_max = Math.max(xm_max, r1.getCharacterization().getXm());
                grad_max = Math.max(grad_max, r1.getCharacterization().getGrad());

                ym_min = Math.min(ym_min, r1.getCharacterization().getYm());
                xm_min = Math.min(xm_min, r1.getCharacterization().getXm());
                grad_min = Math.min(grad_min, r1.getCharacterization().getGrad());
            }
        }

        normalizationValues[0] = xm_max - xm_min;
        normalizationValues[1] = ym_max - ym_min;
        normalizationValues[2] = grad_max - grad_min;

        return normalizationValues;
    }

    /* getter and utility functions */

    public File getSourceFile() {
        return sourceFile;
    }

    public HashMap<TruthTable, Circuit> get() {
        return circuitLibrary;
    }

    public HashMap<LogicType, List<GateRealization>> getRealizations() {
        return gateRealizations;
    }

    public void print() {

        logger.info("Circuit library:");

        for (TruthTable truthTable : circuitLibrary.keySet()) {
            logger.info(truthTable.toString() + " --> " + circuitLibrary.get(truthTable).getIdentifier());
        }

        logger.info("Gate number constraints:");

        for (LogicType type : gateRealizations.keySet()) {
            logger.info(type.name() + ": " + gateRealizations.get(type).size());
        }
    }

    public Integer getNumAvailableGates(LogicType type) {

        if (gateRealizations.get(type) == null)
            return 0;

        return gateRealizations.get(type).size();
    }

    public List<LogicType> getGateTypes() {
        return new ArrayList<>(gateRealizations.keySet());
    }

    public int getFeasibility() {

        OptionalInt feasibility = circuitLibrary.values().stream().mapToInt(c -> c.getExpression().variables().size()).max();

        if (feasibility.isEmpty()) {
            logger.error("Library feasibility could not be determined. Using a default value of 2.");
            return 2;
        } else {
            return feasibility.getAsInt();
        }
    }

    public Double[] getProxWeights() {
        return proxWeights;
    }

    public Double[] getProxNormalization() { return proxNormalization; }

    public Set<String> getGroups() {

        Set<String> groupSet = new HashSet<>();

        for (List<GateRealization> realizationsOfType : gateRealizations.values()) {
            for (GateRealization realization : realizationsOfType) {
                groupSet.add(realization.getGroup());
            }
        }

        return groupSet;
    }

}