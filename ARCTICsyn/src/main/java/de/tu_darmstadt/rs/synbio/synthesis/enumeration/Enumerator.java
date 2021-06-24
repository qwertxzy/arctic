package de.tu_darmstadt.rs.synbio.synthesis.enumeration;

import de.tu_darmstadt.rs.synbio.common.*;
import de.tu_darmstadt.rs.synbio.common.circuit.*;
import de.tu_darmstadt.rs.synbio.common.library.GateLibrary;
import de.tu_darmstadt.rs.synbio.synthesis.util.ExpressionParser;
import de.tu_darmstadt.rs.synbio.synthesis.util.TruthTable;
import org.apache.commons.lang3.StringUtils;
import org.apache.commons.lang3.builder.EqualsBuilder;
import org.apache.commons.lang3.builder.HashCodeBuilder;
import org.jgrapht.Graphs;
import org.logicng.formulas.Formula;
import org.logicng.formulas.Variable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.*;
import java.util.stream.Collectors;

public class Enumerator {

    private static final Logger logger = LoggerFactory.getLogger(Enumerator.class);

    private TruthTable targetTruthTable;
    private final GateLibrary gateLib;
    private final int maxDepth;
    private final int feasibility;
    private final int maxWeight;
    private int weightRelaxation;

    private final int numGroupsInLib;

    private final Circuit templateCircuit;
    private int gateCounter;
    private final List<LogicType> gateTypes;
    private List<PrimitiveCircuit> intermediateCircuits;
    private HashMap<TruthTable, Circuit> resultCircuits;

    private int currentMinSize;

    /* constructor for guided enumeration */

    public Enumerator(GateLibrary gateLib, TruthTable targetTruthTable, int maxDepth, int maxWeight, int weightRelaxation) {

        this(gateLib, targetTruthTable.getSupportSize(), maxDepth, maxWeight);
        this.weightRelaxation = weightRelaxation;
        currentMinSize = Integer.MAX_VALUE - weightRelaxation;

        this.targetTruthTable = targetTruthTable;
    }

    /* constructor for free enumeration */

    public Enumerator(GateLibrary gateLib, int feasibility,  int maxDepth, int maxWeight) {

        this.gateLib = gateLib;
        this.maxDepth = maxDepth;
        this.maxWeight = maxWeight;
        this.feasibility = feasibility;

        this.gateTypes = new ArrayList<>();
        this.gateTypes.add(LogicType.EMPTY);
        this.gateTypes.addAll(gateLib.getGateTypes());

        this.numGroupsInLib = gateLib.getGroups().size();

        // init template circuit
        templateCircuit = new Circuit();
        Gate output = new OutputGate("O");
        templateCircuit.addVertex(output);
    }

    /* helper functions for handling of primitive circuits */

    private List<LogicType> mapEntryToRow(PrimitiveCircuit.Entry entry) {
        return combinations[entry.combId].get(entry.index);
    }

    private boolean isCoveredByLibrary(List<LogicType> row) {

        // check if enough gate groups are available
        int numGates = 0;

        for (LogicType element : row) {
            if (element != LogicType.EMPTY)
                numGates ++;
        }

        if (numGates > numGroupsInLib)
            return false;

        // check availability of gates of each type
        for (LogicType type : gateTypes) {

            if (type == LogicType.EMPTY)
                continue;

            int occurrences = 0;

            for (LogicType gateType : row) {
                if (gateType == type)
                    occurrences++;
            }

            if (occurrences > gateLib.getNumAvailableGates(type))
                return false;
        }

        return true;
    }

    private int getNumberOfInputs(List<LogicType> row) {

        int numInputs = 0;

        for (LogicType gateType : row) {
            if (gateType != LogicType.EMPTY)
                numInputs += gateType.getNumInputs();
        }

        return numInputs;
    }

    private int getNumberOfUnboundInputs(PrimitiveCircuit circuit) {

        // get total number of gate inputs
        int totalInputs = circuit.getList().stream().map(this::mapEntryToRow).mapToInt(this::getNumberOfInputs).sum();

        // get number of gates above level 0 (bound inputs)
        int boundInputs = 0;

        for (int level = 1; level < circuit.getDepth(); level ++) {

            PrimitiveCircuit.Entry entry = circuit.getEntry(level);

            for(LogicType gateType : mapEntryToRow(entry)) {
                if (gateType != LogicType.EMPTY)
                    boundInputs ++;
            }
        }

        return totalInputs - boundInputs;
    }

    private int getCircuitWeight(PrimitiveCircuit circuit) {

        int weight = 0;

        for (PrimitiveCircuit.Entry entry : circuit.getList()) {

            for (LogicType gateType : mapEntryToRow(entry)) {
                weight += gateType.getWeight();
            }
        }

        return weight;
    }

    private boolean hasRedundantInverters(PrimitiveCircuit circuit) {

        for (int level = 0; level < (circuit.getDepth() - 1); level ++) {

            int upperColumn = 0;

            PrimitiveCircuit.Entry entry = circuit.getEntry(level);

            for (LogicType gateType : mapEntryToRow(entry)) {

                if (gateType == LogicType.NOT) {

                    PrimitiveCircuit.Entry upperEntry = circuit.getEntry(level + 1);
                    List<LogicType> upperRow = mapEntryToRow(upperEntry);

                    if (upperRow.get(upperColumn) == LogicType.NOT)
                        return true;
                }

                upperColumn += gateType.getNumInputs();
            }
        }
        return false;
    }

    private boolean isNotEmpty(List<LogicType> row) {

        for (LogicType type : row) {
            if (type != LogicType.EMPTY)
                return true;
        }

        return false;
    }

    /* main algorithm */

    List<List<LogicType>>[] combinations;

    public void enumerate() {

        //logger.info("starting circuit enumeration.");

        intermediateCircuits = new ArrayList<>();
        resultCircuits = new HashMap<>();
        gateCounter = 0;

        OptionalInt maxGateFeasibility = gateTypes.stream().mapToInt(LogicType::getNumInputs).max();
        if (maxGateFeasibility.isEmpty())
            return;

        // max number of gates in uppermost level
        int maxRowLength = (int) Math.pow(maxGateFeasibility.getAsInt(), maxDepth - 1);

        // array of array lists of arrays containing combinations
        combinations = new ArrayList[maxRowLength];

        for (int i = 0; i < maxRowLength; i ++) {
            combinations[i] = new ArrayList<>();
        }

        // generate rows (combinations of gates and empty slots)
        for (int rowLength = 1; rowLength <= maxRowLength; rowLength ++) {

            List<List<LogicType>> lengthCombinations = combinations[rowLength - 1];

            for (Integer i = 0; i < (int) Math.pow(gateTypes.size(), rowLength); i ++) {

                String combination = Integer.toString(i, gateTypes.size());
                combination = StringUtils.leftPad(combination, rowLength, '0');

                ArrayList<LogicType> row = new ArrayList<>(rowLength);

                for (int j = 0; j < combination.length(); j++) {
                    row.add(j, gateTypes.get(Character.getNumericValue(combination.charAt(j))));
                }

                if (isCoveredByLibrary(row) && isNotEmpty(row))
                    if (!(rowLength != 1 && row.contains(LogicType.OR2))) // limit OR gate to output row
                        lengthCombinations.add(row);
            }
        }

        // call recursive circuit build function
        PrimitiveCircuit emptyCircuit = new PrimitiveCircuit(maxDepth);
        buildCircuits(emptyCircuit, 0);

        //logger.info("found " + intermediateCircuits.size() + " pre-filtered circuits.");

        filterRedundantCircuits();

        //logger.info("found " + intermediateCircuits.size() + " structurally different circuits.");

        for (PrimitiveCircuit circuit : intermediateCircuits) {
            buildWireAddCircuit(circuit);
        }
    }

    private void buildCircuits(PrimitiveCircuit circuit, int level) {

        // if max depth reached --> abort
        if (level >= maxDepth)
            return;

        // if circuit is empty --> build start circuits and recurse
        if (level == 0) {

            for (int i = 0; i < combinations[0].size(); i++) {
                PrimitiveCircuit newCircuit = new PrimitiveCircuit(circuit);
                newCircuit.insertEntry(0, i);
                buildCircuits(newCircuit, 1);
            }

        // if circuit is not empty --> extend by next row
        } else {

            // get number of inputs of lower level
            PrimitiveCircuit.Entry entry = circuit.getEntry(level - 1);
            int numberOfInputs = getNumberOfInputs(mapEntryToRow(entry));

            // iterate over rows with corresponding number of gates/entries
            for (int i = 0; i < combinations[numberOfInputs - 1].size(); i++) {

                if (combinations[numberOfInputs - 1].get(i).contains(LogicType.OR2)) // limit OR gate to output row
                    continue;

                PrimitiveCircuit newCircuit = new PrimitiveCircuit(circuit);
                newCircuit.addEntry(level, numberOfInputs - 1, i);

                if (circuitAllowedByPreFilter(newCircuit))
                    intermediateCircuits.add(newCircuit);

                buildCircuits(newCircuit, level + 1);
            }
        }
    }

    private boolean circuitAllowedByPreFilter(PrimitiveCircuit circuit) {

        // check if circuit is implementable with library
        if (!isCoveredByLibrary(circuit.getList().stream().map(this::mapEntryToRow).flatMap(Collection::stream).collect(Collectors.toCollection(ArrayList::new))))
            return false;

        // check is feasibility is met
        if (getNumberOfUnboundInputs(circuit) < feasibility)
            return false;

        // check circuit weight. if it exceeds maximum --> continue
        if (getCircuitWeight(circuit) > maxWeight)
            return false;

        // check if circuit contains redundant inverters
        if (hasRedundantInverters(circuit))
            return false;

        return true;
    }

    private List<HashMap<Coordinates, List<List<LogicType>>>> pathDB;

    private void filterRedundantCircuits() {

        // calculate all paths
        pathDB = new ArrayList<>();

        for (PrimitiveCircuit circuit : intermediateCircuits) {
            pathDB.add(getAllPaths(circuit));
        }

        // filter structurally equivalent circuits
        List<Integer> resultCircuitIndices = new ArrayList<>();
        resultCircuitIndices.add(0);

        for (int i = 0; i < intermediateCircuits.size(); i ++) {

            boolean isUnique = true;

            for (Integer cmpIndex : resultCircuitIndices) {

                if (structurallyEquivalent(intermediateCircuits.get(i), i, intermediateCircuits.get(cmpIndex), cmpIndex)) {
                    isUnique = false;
                    break;
                }
            }

            if (isUnique) {
                resultCircuitIndices.add(i);
                //logger.info("circuit " + i + " added");
            }
        }

        // build list of filtered circuits
        List<PrimitiveCircuit> filteredCircuits = new ArrayList<>();

        for (Integer index : resultCircuitIndices) {
            filteredCircuits.add(intermediateCircuits.get(index));
        }

        intermediateCircuits = filteredCircuits;
    }

    private boolean structurallyEquivalent(PrimitiveCircuit circuit1, int circuit1Index, PrimitiveCircuit circuit2, int circuit2Index) {

        if (circuit1.getDepth() != circuit2.getDepth())
            return false;

        if (getCircuitWeight(circuit1) != getCircuitWeight(circuit2))
            return false;

        // get paths
        HashMap<Coordinates, List<List<LogicType>>> paths1 = pathDB.get(circuit1Index);
        HashMap<Coordinates, List<List<LogicType>>> paths2 = pathDB.get(circuit2Index);

        // create mark list
        List<List<Boolean>> circuit2Marks = new ArrayList<>();

        for (PrimitiveCircuit.Entry entry : circuit2.getList()) {

            ArrayList<Boolean> markRow = new ArrayList<>();
            circuit2Marks.add(markRow);

            for (LogicType ignored : mapEntryToRow(entry)) {
                markRow.add(false);
            }
        }

        // iterate over elements of circuit 1 and search equivalents in circuit 2
        for (int level = 0; level < circuit1.getDepth(); level ++) {

            if (mapEntryToRow(circuit1.getEntry(level)).size() != mapEntryToRow(circuit2.getEntry(level)).size())
                return false;

            int numColumns = mapEntryToRow(circuit1.getEntry(level)).size();

            for (int column = 0; column < numColumns; column ++) {

                LogicType element = mapEntryToRow(circuit1.getEntry(level)).get(column);
                Coordinates coords1 = new Coordinates(level, column);
                List<List<LogicType>> element1Paths = paths1.get(coords1);

                boolean equivalentFound = false;

                // search equivalent gate in circuit 2
                for (int column2 = 0; column2 < numColumns; column2 ++) {

                    // if element is same type and is not yet marked
                    if (mapEntryToRow(circuit2.getEntry(level)).get(column2) == element && !circuit2Marks.get(level).get(column2)) {

                        Coordinates coords2 = new Coordinates(level, column2);
                        List<List<LogicType>> element2Paths = paths2.get(coords2);

                        if (element2Paths.containsAll(element1Paths) && element1Paths.containsAll(element2Paths)) {
                            circuit2Marks.get(level).set(column2, true);
                            equivalentFound = true;
                            break;
                        }
                    }
                }

                if (!equivalentFound)
                    return false;
            }
        }

        // check if every element of circuit 2 is marked
        for (List<Boolean> level : circuit2Marks) {
            for (Boolean marked : level) {
                if (!marked)
                    return false;
            }
        }

        return true;
    }

    private HashMap<Coordinates, List<List<LogicType>>> getAllPaths(PrimitiveCircuit circuit) {

        HashMap<Coordinates, List<List<LogicType>>> paths = new HashMap<>();

        for (int level = circuit.getDepth() - 1; level >= 0; level --) {

            int rowLength = mapEntryToRow(circuit.getEntry(level)).size();

            for (int column = 0; column < rowLength; column ++) {

                Coordinates currentNode = new Coordinates(level, column);
                List<Coordinates> connectedNodes = getInputNodes(circuit, currentNode);

                paths.putIfAbsent(currentNode, new ArrayList<>());

                if (connectedNodes.isEmpty()) {

                    ArrayList<LogicType> path = new ArrayList<>();
                    path.add(mapEntryToRow(circuit.getEntry(level)).get(column));
                    paths.get(currentNode).add(path);
                }

                for (Coordinates node : connectedNodes) {

                    for (List<LogicType> path : paths.get(node)) {

                        ArrayList<LogicType> extendedPath = new ArrayList<>(path);
                        extendedPath.add(mapEntryToRow(circuit.getEntry(level)).get(column));
                        paths.get(currentNode).add(extendedPath);
                    }

                }
            }
        }

        return paths;
    }

    private List<Coordinates> getInputNodes(PrimitiveCircuit circuit, Coordinates coords) {

        List<Coordinates> inputNodes = new ArrayList<>();

        LogicType element = mapEntryToRow(circuit.getEntry(coords.level)).get(coords.column);
        int numInputs = element.getNumInputs();

        // if element is on uppermost level --> return empty list
        if (coords.level == (circuit.getDepth() - 1)) {
            return inputNodes;
        }

        int upperInput = 0;

        for (int currentColumn = 0; currentColumn < coords.column; currentColumn ++) {

            LogicType currentElement = mapEntryToRow(circuit.getEntry(coords.level)).get(currentColumn);

            if (currentElement != LogicType.EMPTY)
                upperInput += currentElement.getNumInputs();
        }

        for (int input = upperInput; input < (upperInput + numInputs); input ++) {

            inputNodes.add(new Coordinates(coords.level + 1, input));
        }

        return inputNodes;

    }

    static private class Coordinates {

        public final Integer level;
        public final Integer column;

        public Coordinates(Integer level, Integer column) {
            this.level = level;
            this.column = column;
        }

        @Override
        public boolean equals(Object o) {
            if (!(o instanceof Coordinates)) {
                return false;
            }
            Coordinates cmp  = (Coordinates) o;
            EqualsBuilder builder = new EqualsBuilder();
            builder.append(level, cmp.level);
            builder.append(column, cmp.column);
            return builder.isEquals();
        }

        @Override
        public int hashCode() {
            HashCodeBuilder builder = new HashCodeBuilder();
            builder.append(level);
            builder.append(column);
            return builder.hashCode();
        }
    }

    private void buildWireAddCircuit(PrimitiveCircuit candidate) {

        /* build circuit */
        Circuit circuit = new Circuit();
        Graphs.addGraph(circuit, templateCircuit);

        // add first logic gate after output
        LogicType type = mapEntryToRow(candidate.getEntry(0)).get(0);
        gateCounter = 0;
        Gate newGate = new LogicGate(type.name() + "_" + gateCounter, type);
        circuit.addVertex(newGate);
        circuit.addEdge(newGate, circuit.getOutputBuffer(), new Wire(circuit.getOutputBuffer().getExpression().variables().first()));

        // add rows of gates and wire them
        List<Gate> prevRow = new ArrayList<>();
        List<Gate> currentRow = new ArrayList<>();
        prevRow.add(newGate);

        for (int level = 1; level < candidate.getDepth(); level ++) {

            int column = 0;

            for (Gate anchorGate : prevRow) {

                for (Variable anchorVariable : anchorGate.getExpression().variables()) {

                    type = mapEntryToRow(candidate.getEntry(level)).get(column);

                    if (type != LogicType.EMPTY) {

                        gateCounter++;
                        newGate = new LogicGate(type.name() + "_" + gateCounter, type);
                        circuit.addVertex(newGate);
                        circuit.addEdge(newGate, anchorGate, new Wire(anchorVariable));

                        currentRow.add(newGate);
                    }

                    column++;
                }
            }

            prevRow = new ArrayList<>(currentRow);
            currentRow.clear();
        }

        /* wire inputs, post-build checks, addition */
        wireAndAddCircuit(circuit);
    }

    private void wireAndAddCircuit(Circuit candidate) {

        Circuit dummyCandidate = new Circuit();
        Graphs.addGraph(dummyCandidate, candidate);

        // list ports to be connected
        HashMap<Gate, List<Variable>> gateInputs = dummyCandidate.getUnconnectedGateInputs();

        char inputName = 'a';
        for (Gate gateWithInput : gateInputs.keySet()) {
            for (Variable var : gateInputs.get(gateWithInput)) {
                InputGate inputGate = new InputGate(ExpressionParser.parse(Character.toString(inputName)), Character.toString(inputName));
                dummyCandidate.addVertex(inputGate);
                dummyCandidate.addEdge(inputGate, gateWithInput, new Wire(var));
                inputName ++;
            }
        }

        Formula dummyExpression = dummyCandidate.getExpression();

        List<String> inputMappings = getInputMappings(dummyExpression, feasibility);

        building:
        for (String mapping : inputMappings) {

            // wire according to mapping
            Circuit wiredCircuit = new Circuit("gen_circuit");
            Graphs.addGraph(wiredCircuit, candidate);

            List<InputGate> inputGates = new ArrayList<>();
            char inputGateName = 'a';
            for (int i = 0; i < feasibility; i ++) {
                inputGates.add(new InputGate(ExpressionParser.parse(Character.toString(inputGateName)), Character.toString(inputGateName)));
                inputGateName ++;
            }

            int gateInputIdx = 0;
            for (Gate gateWithInput : gateInputs.keySet()) {
                for (Variable var : gateInputs.get(gateWithInput)) {

                    InputGate inputGate = inputGates.get(Integer.parseInt(String.valueOf(mapping.charAt(gateInputIdx))));
                    if (!wiredCircuit.containsVertex(inputGate))
                        wiredCircuit.addVertex(inputGate);

                    //check if same input is wired multiple times to one gate --> abort
                    if (wiredCircuit.containsEdge(inputGate, gateWithInput))
                        continue building;

                    wiredCircuit.addEdge(inputGate, gateWithInput, new Wire(var));
                    gateInputIdx ++;
                }
            }

            // remove redundant gates
            if (!wiredCircuit.removeRedundantGates())
                continue;

            // compare to current min size
            int weight = wiredCircuit.getWeight();
            if (weight > currentMinSize + weightRelaxation)
                continue;

            TruthTable truthTable = new TruthTable(wiredCircuit.getExpression());

            // filter out structurally equivalent circuits
            for (TruthTable libTruthTable : resultCircuits.keySet()) {

                if (targetTruthTable != null || libTruthTable.equalsLogically(truthTable)) {

                    if (wiredCircuit.isEquivalent(resultCircuits.get(libTruthTable))) {
                        continue building;
                    }
                }
            }

            if (weight < currentMinSize) {
                currentMinSize = weight;

                // update result circuits
                resultCircuits.entrySet().removeIf(e -> e.getValue().getWeight() > (currentMinSize + weightRelaxation));
            }

            resultCircuits.put(truthTable, wiredCircuit);
        }
    }

    private List<String> getInputMappings(Formula rawExpression, int feasibility) {

        List<TruthTable> truthTables = new ArrayList<>();
        List<String> inputMappings = new ArrayList<>();

        List<Variable> rawExpressionVars = new ArrayList<>(rawExpression.variables());
        int numRawExpressionVariables = rawExpressionVars.size();

        if (numRawExpressionVariables < feasibility)
            return Collections.emptyList();

        List<Formula> inputVars = new ArrayList<>();
        char varName = 'a';
        for (int i = 0; i < feasibility; i ++) {
            inputVars.add(ExpressionParser.parse(varName + "i"));
            varName ++;
        }

        for (Integer i = 0; i < (int) Math.pow(feasibility, numRawExpressionVariables); i ++) {

            String combination = Integer.toString(i, feasibility);
            combination = StringUtils.leftPad(combination, numRawExpressionVariables, '0');

            // test if mapping contains all primary inputs
            boolean valid = true;
            for (int j = 0; j < feasibility; j++) {
                if (combination.indexOf(Character.forDigit(j, feasibility)) == -1) {
                    valid = false;
                    break;
                }
            }
            if (!valid)
                continue;

            // substitute vars according to mapping
            Formula expression = rawExpression;

            for (int j = 0; j < combination.length(); j++) {

                Formula substitution = inputVars.get(Character.getNumericValue(combination.charAt(j)));
                Variable varToSubstitute = rawExpressionVars.get(j);

                expression = expression.substitute(varToSubstitute, substitution);
            }

            TruthTable tt = new TruthTable(expression);

            if (targetTruthTable != null && targetTruthTable.equalsLogically(tt)) {
                inputMappings.add(combination);
                //return inputMappings;
            } else if (targetTruthTable == null) {

                boolean newMappingFound = true;

                for(TruthTable cmp : truthTables) {
                    if (cmp.equalsLogically(tt)) {
                        newMappingFound = false;
                        break;
                    }
                }

                if (newMappingFound) {
                    inputMappings.add(combination);
                    truthTables.add(tt);
                }
            }
        }

        return inputMappings;
    }

    /* getter */

    public HashMap<TruthTable, Circuit> getResultCircuits() {
        return resultCircuits;
    }
}
