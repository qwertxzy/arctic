# -*- coding: utf-8 -*-
import json
import parser
import multiprocessing
import ast
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import ot
import time

import SimulatorEnvelopExtension as see
import SimulatorVisualisationExtension as sve

"""
Created on Thu Jan 16 18:41:41 2020

@author: Erik Kubaczka
@author: Nicolai Engelmann
"""

"""
    !!!
    Attention: The proposed internal circuit representation can only be used for combinational circuits
    !!!
"""

VERSION_NUMBER = "3.1.2"


# Returns the content of a .json file at the provided path
def loadJSON(filePath):
    with open(filePath, 'r') as jsonFile:
        data = json.load(jsonFile)
    return data


# A wrapper for plotting data in logHist scale
def logHistWrapper(data):
    logHist(data=data, xmin=10 ** -5, xmax=10 ** 3)  # , ymin = 0.0001, ymax = 10)


# Plots the data in a histogram with a logarithmic x-scale
def logHist(data, xmin, xmax):  # , ymin, ymax):
    bins = np.logspace(np.log10(xmin), np.log10(xmax), 120)
    fig = plt.figure()
    plt.hist(data, bins)
    plt.gca().set_xscale('log')
    plt.show();


# Wrapper for plotting two data sets in a x-scale log Histogram
def logHistWrapper2(data1, data2):
    logHist2(data1=data1, data2=data2, xmin=10 ** -5, xmax=10 ** 3)  # , ymin = 0.0001, ymax = 10)


# Plots two data sets in a histogram with logarithmic x-scale
def logHist2(data1, data2, xmin, xmax):  # , ymin, ymax):
    bins = np.logspace(np.log10(xmin), np.log10(xmax), 120)
    fig = plt.figure()
    plt.hist(data1, bins)
    plt.hist(data2, bins)
    plt.gca().set_xscale('log')
    plt.show();


# Wrapper for logHistSubplot
def logHistWrapperSubplot(data1, data2):
    logHistSubplot(data1=data1, data2=data2, xmin=10 ** -5, xmax=10 ** 3)  # , ymin = 0.0001, ymax = 10)


# Plots a hist with log x-scale as a subplot
def logHistSubplot(data1, data2, xmin, xmax):  # , ymin, ymax):
    # print(data1, data2)
    bins = np.logspace(np.log10(xmin), np.log10(xmax), 120)
    fig = plt.figure()
    ax = plt.subplot(211)
    ax.title.set_text("Lowest ON")
    plt.hist(data1, bins)
    plt.gca().set_xscale('log')

    ax = plt.subplot(212)  # , xscale="log")
    ax.title.set_text("Highest OFF")
    plt.hist(data2, bins)
    plt.gca().set_xscale('log')
    plt.draw()
    plt.show()


# Returns the Boolean input combination in dependence of the value (val) to represent and the number of inputs of the circuit
def getCombination(val, numberOfInputs):
    combination = bin(val).replace("0b", "")
    missingZeros = numberOfInputs - len(combination)
    for i in range(missingZeros):
        combination = "0" + combination

    return combination


# Maps a Boolean input assignment to the corresponding concetrations representing this assignment
def getBioVals(inputIDs, combination, inputSpecification, envelopMode=False):
    i = 0
    bioVals = {}
    for inputID in inputIDs:
        if (envelopMode):
            bioVals[inputID] = inputSpecification[inputID][str(combination[int(i)])]
            i += 0.5  # Increment only by 0.5, since in envelopMode two inputs correspond to the same boolean state
        else:
            bioVals[inputID] = inputSpecification[inputID][str(combination[int(i)])]
            i += 1  # Increment by one, to get the next entries boolean value.

    return bioVals


# Creates a .pkl including the truthTable data,
# the simulation results as well as the over all circuit values and the arguments passed to this script
def saveSimResults(truthTable, simRes, completeCircuitVals=[]):
    suffix = simContext["store_suffix"]
    fileName = "simulation_results_" + suffix + ".pkl"
    with open(fileName, "wb") as file:
        pickle.dump({"truthtable": truthTable,
                     "simulation_results": simRes,
                     "complete_circuit_vals": completeCircuitVals,
                     "sym_args": sys.argv}, file)


# Calculates the circuit score based on the data provided for ON (dataON) and OFF (dataOFF)
def circuitScore(dataON, dataOFF):
    median_on = np.median(dataON)
    median_off = np.median(dataOFF)
    sign = 1 if np.median(dataON) > np.median(
        dataOFF) else -1  # Determine the sign of the exponent based on the median differences
    score = 0;
    DIST = simContext["dist"]
    if ("kl" == DIST):  # Minimum of Kullback-Leibler Divergence
        mu_1 = np.mean(dataON)
        var_1 = np.var(dataON)
        mu_2 = np.mean(dataOFF)
        var_2 = np.var(dataOFF)

        score = min([0.5 * np.log(var_2 / var_1) + ((mu_1 - mu_2) ** 2 + var_1 - var_2) / (2 * var_2),
                     0.5 * np.log(var_1 / var_2) + ((mu_2 - mu_1) ** 2 + var_2 - var_1) / (2 * var_1)]);
    elif ("ws" == DIST):  # Wasserstein Distance
        # score = sign * st.wasserstein_distance(dataON, dataOFF)
        score = sign * ot.wasserstein_1d(dataON, dataOFF, p=simContext["wasserstein_p"])
    elif ("cello" == DIST):  # To prefer as Cello Score
        score = min(dataON) / max(dataOFF)
    elif ("ws-cello" == DIST):
        score = sign * ot.wasserstein_1d(dataON, dataOFF, p=simContext["wasserstein_p"])
        score = sign * score / np.median(dataOFF) + 1
    elif ("ws-cello-m" == DIST):
        score = sign * ot.wasserstein_1d(dataON, dataOFF, p=simContext["wasserstein_p"])
        score = score / np.mean(dataOFF) + 1
    elif ("ws-log" == DIST):
        score = sign * ot.wasserstein_1d(np.log(dataON), np.log(dataOFF), p=simContext["wasserstein_p"])
    elif ("ws-exp" == DIST):
        score = sign * ot.wasserstein_1d(np.exp(dataON), np.exp(dataOFF), p=simContext["wasserstein_p"])
    elif ("ws-log-exp" == DIST):  # To prefer as WS Score
        # score = sign * st.wasserstein_distance(np.log(dataON), np.log(dataOFF))
        score = sign * ot.wasserstein_1d(np.log(dataON), np.log(dataOFF), p=simContext["wasserstein_p"])
        score = np.exp(score)
    elif ("ws-log-exp-asym" == DIST):
        score = sign * ot.wasserstein_1d(np.log(dataON[dataON <= median_on]), np.log(dataOFF[dataOFF >= median_off]),
                                         p=simContext["wasserstein_p"])
        score = np.exp(score)
    else:
        score = -9999  # (np.mean(dataON) - np.mean(dataOFF)) ** 2    # Error Code if no appropriate score is used
    return score;


"""
This method is invoked to prepare the data relevant for the simulation.
This includes the creation of a suitable circuit representation, the preparation of the response curves as well as the particles used for simulation
and transformations required for calculating the envelop model based on the aequivalent envelop free model.

structureContent: The data included in the provided structure file
gateLib: The library of genetic gates which shall be used for simulation
maxNumberOfParticles: The maximum number of particles applicable to use during simulation.
"""""


def initialiseSimulator(structureContent, gateLib, maxNumberOfParticles):
    # The information in the genetic gate library is transformed into a suitable representation adequate for the subsequent simulation
    def generateResponseFunctions(gateLib):
        def load_expression(source_string):
            st = parser.expr(source_string)
            return st.compile()

        envelopMode = simContext["envelop_mode"]
        if (envelopMode):
            responseFunctions = see.parseEnvelopeLibToGateLib(gateLib=gateLib)
        else:
            responseFunctions = {gate['identifier']: 0 for gate in gateLib}

            for gate in gateLib:
                if ((gate["biorep"]["response_function"]["type"] == "INHIBITORY_HILL_EQUATION" or "IMPLICIT_OR" ==
                     gate["biorep"]["response_function"]["type"]) and True):
                    # If native, the simulator makes use of the built in functions and does not evaluate the the given code for not natively supported expressions

                    responseFunctions[gate["identifier"]] = {"native": True,
                                                             "type": gate["biorep"]["response_function"]["type"]}
                else:
                    responseFunctions[gate["identifier"]] = {"native": False, "expression": load_expression(
                        gate["biorep"]["response_function"]["equation"])}

                responseFunctions[gate["identifier"]]["parameters"] = dict.copy(
                    gate["biorep"]["response_function"]["parameters"])
                responseFunctions[gate["identifier"]]["library_parameters"] = dict.copy(
                    gate["biorep"]["response_function"]["parameters"])

                if ("particles" in gate["biorep"]):
                    responseFunctions[gate["identifier"]]["particle_parameters"] = list(
                        gate["biorep"]["particles"].keys())

        return responseFunctions

    """
    This method prepares the particles for the gates  
    The particles provided in the gatelib are used and filled up if their number is not sufficient for the maximum number of particles required.
    
    gateLib: The genetic gate library to use
    n: The maximum number of particles applicable to use during simulation
    """

    def prepareParameters(gateLib, n):
        def generateParticles(k, n):
            if (n > 1):
                sigma_k = 0.8
                mu_k = np.log(k) - 0.5 * sigma_k ** 2
                return np.random.lognormal(mu_k, sigma_k, n)
            else:
                return k

        if (n < 0):
            raise Exception('The number of necessary particles should be greater than zero!')

        # Create the empty particles lib
        particlesLib = {"max_number_of_particles": n}

        # In case of envelop mode, a different treatment is necessary
        if (simContext["envelop_mode"]):
            see.prepareEnvelopeParameters(particlesLib, gateLib, n)
            return particlesLib

        for gate in gateLib:
            # The gates entry needs to include a particles entry. Also empty entries are possible.
            # Relevant for accounting the implicit or which is not able to feature particles
            if ("particles" in gate["biorep"]):
                # Add placeholder for particles of Gate "gate"
                particlesLib[gate['identifier']] = {}

                # Obtain all parameters for which particles are provided or shall be provided
                for parameter in gate['biorep']['particles']:
                    # Get the particles included in the library
                    rawParticles = gate['biorep']['particles'][parameter]

                    numberOfProvidedParticles = len(rawParticles)
                    if (numberOfProvidedParticles > 0):

                        # Generate additional particles, if the number of particles provided is smaller then the number of particles requested
                        if (numberOfProvidedParticles < n):
                            # The missing particles are generated based on a log normal, whereby the mean of the provided particles is taken into account
                            additionalParticles = generateParticles(np.mean(rawParticles),
                                                                    n - numberOfProvidedParticles)
                        # The number of considered particles is limited to the required maximum number
                        if (numberOfProvidedParticles > n):
                            numberOfProvidedParticles = n

                        particles = np.zeros(n)

                        for iX in range(numberOfProvidedParticles):  # Add all provided particles
                            particles[iX] = rawParticles[iX]

                        for iX in range(numberOfProvidedParticles, n):  # Add the additionally generated particles
                            particles[iX] = additionalParticles[iX - numberOfProvidedParticles]

                    else:
                        # Directly generate the particles, whereby the parameters value is taken as mean
                        particles = generateParticles(gate['biorep']['response_function']["parameters"][parameter], n)

                    # Assign the particles
                    particlesLib[gate['identifier']][parameter] = particles

        return particlesLib

    #   This method checks whether the circuit is represented in the right order,
    #   thus the gates are in the order required for the calculation.
    # This order sufficies the topological order
    def isValid(nodes, edges):
        i = 0
        for node in nodes:
            sources = getSources(edges, node['id'])

            for source in sources:
                # print(source, nodes[i:] )
                if (source in [nod['id'] for nod in nodes[i:]]):
                    return False
            i += 1

        return True

    """
    This method returns a list of node IDs, which is sorted in the order of evaluation for the simulation.
    
    nodes: The nodes of the circuit
    edges: The edges of the circuit
    """

    def createValidNodeIDList(nodes, edges):
        # Checks if all sources the node nodeID is connected to via edges is included in validNodeIDs
        def allSourcesPresent(edges, nodeID, validNodeIDs):
            targetSources = getSources(edges, nodeID)
            for targetSource in targetSources:
                if (not targetSource in validNodeIDs):
                    return False
            return True

        validNodeIDs = []

        # Obtains a dict including the nodes corresponding to INPUTS, OUTPUTS and LOGIC Gates
        nodeSets = getSets(nodes)
        sortedInputs = sorted(nodeSets['INPUT'], key=lambda d: d['id'])  # Sorts the input Nodes by ID

        # Append the inputs to the set of validNodeIDs
        for inVal in sortedInputs:
            validNodeIDs.append(inVal['id'])

        i = 0
        while (len(validNodeIDs) != len(
                nodes)):  # As long as their are not all nodes included in the set of valid node ids continue
            targets = getTargets(edges, validNodeIDs[i])  # Get the targets of the next node in validNodeIDs
            for target in targets:
                if (
                        not target in validNodeIDs):  # Add each target to the list in case it is not already included and all sources are present.
                    if (allSourcesPresent(edges, target, validNodeIDs)):
                        validNodeIDs.append(target)

            i += 1

        return validNodeIDs

    def sortNodes(validNodeIDs, nodeDict):
        validNodes = []
        for validNodeID in validNodeIDs:
            validNodes.append(nodeDict[validNodeID])

        return validNodes

    def getNodeDict(nodes):
        nodeDict = {node['id']: node for node in nodes}
        return nodeDict

    # Separates the nodes in INPUT, LOGIC and OUTPUT gates
    def getSets(nodes):
        graphSet = {"INPUT": [], "LOGIC": [], "OUTPUT": []}
        for node in nodes:
            graphSet[node['type']].append(node)

        return graphSet

    def getTargets(edges, sourceID):
        targets = []
        for edge in edges:
            if (edge['source'] == sourceID):
                targets.append(edge['target'])
        return targets

    def getSources(edges, targetID):
        sources = []
        for edge in edges:
            if (edge['target'] == targetID):
                sources.append(edge['source'])
        return sources

    def generateCircuit(nodeDict, edges, validNodeIDs):
        circuit = {nodeID: getTargets(edges, nodeID) for nodeID in validNodeIDs}
        return circuit

    """
    This method generates the truthtable essential for simulation.
    This truthtable includes the boolean as well as bio assignments and the boolean output values of the circuit.
    These information is essential for the simulation as well as the subsequent scoring.
    
    inputIDs: The ids of the input buffers
    outputIDs: The ids of the output buffers
    outputTruthTable: The truthtable string of the boolean circuit
    inputSpecification: Includes the on and off concentrations for each input buffer
    envelopMode: Indicates whether the envelope mode needs to be used or not
    whiteListString: String indicating which input assignments to consider and which not (required for subproblem simulation)
    substitutionList: Indicates which values needs to be substituted and which not (required for subproblem simulation)
    substitutionTruthtables: Set of truthtables for the nodes whose outputs need to be substituted (required for subproblem simulation)
    """

    def generateTruthTable(inputIDs, outputIDs, outputTruthTable, inputSpecification, envelopMode=False,
                           whiteListString=None, substitutionsList=None, substitutionTruthtables=None):

        sortedInputIDs = sorted(inputIDs)
        sortedOutputIDs = sorted(outputIDs)

        inputIDs = {sortedInputIDs[val]: val for val in range(len(sortedInputIDs))}
        outputIDs = {sortedOutputIDs[val]: val for val in range(len(sortedOutputIDs))}

        # Prepare the final truthtable
        truthTable = {"input_IDs": inputIDs,
                      "output_IDs": outputIDs,
                      "inputs": {},  # {"input_" + str(iVal): [] for iVal in range(len(outputTruthTable))},
                      "bio_inputs": {},  # {"input_" + str(iVal): [] for iVal in range(len(outputTruthTable))},
                      "outputs": {},  # {"output_" + str(iVal): [] for iVal in range(len(outputTruthTable))}
                      "substitutions": {},
                      "substitution_truthtables": {key: "" for key in substitutionTruthtables} if (
                              substitutionTruthtables != None) else {},
                      # It is not possible to directly copy the substitutionsTruthtable due to whitelisting
                      "input_specification": inputSpecification
                      }

        numberOfInputs = len(inputIDs)
        if (envelopMode):  # Consistency check for envelop mode
            if (numberOfInputs % 2 != 0):
                raise Exception("Some problem in creation of inputIDs")
            numberOfInputs = int(numberOfInputs / 2)

        for i in range(len(outputTruthTable)):
            if (whiteListString != None and whiteListString[
                i] == "0"):  # If the assignment is not positively included in the whitelist, ignore it
                continue
            combination = getCombination(i, numberOfInputs)  # Translate integer value to boolean combination
            inputIdentString = "input_" + str(i)
            truthTable["inputs"][inputIdentString] = [int(c) for c in combination]
            identString = "output_" + str(i)
            truthTable["outputs"][identString] = {}  # Only works for just one circuit output
            for outID in sortedOutputIDs:
                truthTable["outputs"][identString][outID] = outputTruthTable[i]

            truthTable["bio_inputs"][inputIdentString] = getBioVals(inputIDs, combination, inputSpecification,
                                                                    envelopMode)

            # Add substitution information if present
            if (substitutionsList != None):
                truthTable["substitutions"][inputIdentString] = substitutionsList[i]
            else:
                truthTable["substitutions"][inputIdentString] = {}

            if (substitutionTruthtables != None):
                for key in substitutionTruthtables:
                    truthTable["substitution_truthtables"][key] += substitutionTruthtables[key][i]

        return truthTable

    # Specifies a basic input specification
    def generateInputSpecification(inputIDs, inLow=0.01, inHigh=3):
        inputSpecification = {inID: {"0": inLow, "1": inHigh} for inID in inputIDs}
        return inputSpecification

    # Specifies an input specification according to cellos input specification.
    # Every other input buffer is assigned the same concentration
    def generateCelloInputSpecification(inputIDs):
        inputSpecification = {inputID: {"0": 0.0082, "1": 2.5} for inputID in inputIDs}
        # Overwrites Cello specific input IDs with Cello specific values.
        inputSpecification["a"] = {"0": 0.0034, "1": 2.8}
        inputSpecification["b"] = {"0": 0.0013, "1": 4.4}
        inputSpecification["c"] = {"0": 0.0082, "1": 2.5}
        inputSpecification["d"] = {"0": 0.025, "1": 0.31}
        return inputSpecification

    # Cello input specification for envelop mode.
    def generateCelloEnvelopeInputSpecification(inputIDs):
        inputSpecification = {inputID: {"0": 0.0082, "1": 2.5} for inputID in inputIDs}
        # Overwrites Cello specific input IDs with Cello specific values.
        inputSpecification["a_H"] = {"0": 0.0034 * 2, "1": 2.8}
        inputSpecification["a_L"] = {"0": 0.0034, "1": 2.8 / 2}
        inputSpecification["b_H"] = {"0": 0.0013 * 2, "1": 4.4}
        inputSpecification["b_L"] = {"0": 0.0013, "1": 4.4 / 2}
        inputSpecification["c_H"] = {"0": 0.0082 * 2, "1": 2.5}
        inputSpecification["c_L"] = {"0": 0.0082, "1": 2.5 / 2}
        inputSpecification["d_H"] = {"0": 0.025 * 2, "1": 0.31}
        inputSpecification["d_L"] = {"0": 0.025, "1": 0.31 / 2}
        return inputSpecification

    # Envelop mode input specification.
    def generateEnvelopeInputSpecification(inputIDs):
        inputSpecification = {}
        for key in inputIDs:
            keyLen = len(key)
            if (key[keyLen - 2:] == "H"):
                inputSpecification[key] = {"0": 0.01, "1": 3}  # Upper Envelope Part
            else:
                inputSpecification[key] = {"0": 0.001, "1": 2}  # Lower Envelope Part

        return inputSpecification

    # extract the contents of the structure file
    nodes = structureContent['graph']['nodes']
    edges = structureContent['graph']['edges']
    nodeDict = getNodeDict(nodes)
    nodeSets = getSets(nodes)

    # Determine the order of nodeids required for simulation
    validNodeIDs = createValidNodeIDList(nodes, edges)
    validNodes = sortNodes(validNodeIDs, nodeDict)

    inputIDs = [node['id'] for node in nodeSets['INPUT']]
    outputIDs = [node['id'] for node in nodeSets['OUTPUT']]

    envelopMode = simContext["envelop_mode"]

    # Generate the required circuit representation based on the nodes, the edges and the required order stored in validNodeIDs
    circuit = generateCircuit(nodeDict, edges, validNodeIDs)

    # Copy the original circuit
    originalCircuit = circuit.copy()
    # If envelop mode shall be used, the equivalent envelop free circuit needs to be determined.
    # generateEnvelopCircuit is a method of the Simulator Envelop Extension
    if (envelopMode):
        envelopeCircuit, envelopeInputIDs, envelopeOutputIDs = see.generateEnvelopCircuit(circuit=circuit,
                                                                                          inputIDs=inputIDs,
                                                                                          outputIDs=outputIDs,
                                                                                          nodeDict=nodeDict)

    # TODO Adapt circuitSimulation, inputIDs and outputIDs to envelopMode
    circuitInfo = {'NUMBER_OF_INPUTS': len(nodeSets['INPUT']),
                   'NUMBER_OF_OUTPUTS': len(nodeSets['OUTPUT']),
                   'NUMBER_OF_LOGICS': len(nodeSets['LOGIC'])}

    # Check if default input specification shall be overwritten.
    if (simContext["use_custom_input_specification"]
            and (("custom_input_low" in simContext
                  and "custom_input_high" in simContext) or simContext["custom_input_specification"] != None)):
        # Overwrites the default input specification for this simulator instance. This persists for every subsequent simulation.
        inputSpecification = simContext["custom_input_specification"]
        if (inputSpecification == None):
            # This method assumes, that the input specification includes explicitly includes the envelop mode input buffers
            inputSpecification = generateInputSpecification(inputIDs if (not envelopMode) else envelopeInputIDs,
                                                            simContext["custom_input_low"],
                                                            simContext["custom_input_high"])
        # Reset in order to enable explicit overwrite during simulation and automatic reconstruction of the initialisation setting if cis=0
        simContext["custom_input_specification"] = False
    else:  # Use default input specifications
        if (envelopMode):
            inputSpecification = generateCelloEnvelopeInputSpecification(envelopeInputIDs)
        else:
            inputSpecification = generateCelloInputSpecification(inputIDs)

    truthTableString = structureContent['truthtable']
    truthTableString = truthTableString[
                       len(truthTableString)::-1]  # Reverse truthtable string, since it is provided in reversed order

    # Check if whiteliststring exists. If not, it is left empty, otherwise it is reversed
    whiteListString = None
    if (simContext["whitelist"] and "whitelist" in structureContent):
        whiteListString = structureContent["whitelist"]
        whiteListString = whiteListString[len(whiteListString)::-1]

    # Check if substitutionslist exists and inserts it if so
    substitutionslist = None
    if ('substitutions_list' in structureContent):
        substitutionslist = structureContent['substitutions_list']

    # Inserts the substitutiontruthtables in case they exist, these also need to be reversed
    substitutionTruthtables = None
    if ('substitution_truthtables' in structureContent):
        substitutionTruthtables = structureContent['substitution_truthtables']
        for key in substitutionTruthtables:
            # Reverse the truthtables
            substitutionTruthtables[key] = substitutionTruthtables[key][len(substitutionTruthtables[key])::-1]

    # The truthtable parameters for the truthtable creation depend on whether envelop mode is used or not.
    # In each case, the corresponding input and output ids are passed.
    if (envelopMode):
        truthTable = generateTruthTable(inputIDs=envelopeInputIDs, outputIDs=envelopeOutputIDs,
                                        outputTruthTable=truthTableString, inputSpecification=inputSpecification,
                                        envelopMode=envelopMode, whiteListString=whiteListString,
                                        substitutionsList=substitutionslist,
                                        substitutionTruthtables=substitutionTruthtables)
        circuit = envelopeCircuit
    else:
        truthTable = generateTruthTable(inputIDs=inputIDs, outputIDs=outputIDs, outputTruthTable=truthTableString,
                                        inputSpecification=inputSpecification, envelopMode=envelopMode,
                                        whiteListString=whiteListString, substitutionsList=substitutionslist,
                                        substitutionTruthtables=substitutionTruthtables)

    # Create empty circuit
    validNodeIDs = list(circuit.keys())
    circuitSimulation = {nodeID: 0 for nodeID in validNodeIDs}
    # gateLib = loadJSON(gateLibPath)
    responseFunctions = generateResponseFunctions(gateLib)

    # Load particles if present or generate new (no differentiation between simulation with or without particles currently)
    # If not enough particles are given, additional ones will be generated according to the mean of provided particles
    responseFunctionParameters = prepareParameters(gateLib, maxNumberOfParticles)

    # Determining the envelop bounding mass in case stepped envelopes are used
    envelopBoundingMass = None
    if (envelopMode):
        envelopBoundingMass = see.prepareEnvelopBoundingMass(gateLib)

    originalInputOutput = {"input_IDs": inputIDs,
                           "output_IDs": outputIDs}

    # Store the prepared data in two fields for the later simulation
    simData = {'VERSION_NUMBER': VERSION_NUMBER,
               'circuit_info': circuitInfo,
               'circuit': circuit,
               'circuit_original': originalCircuit,
               'circuit_valid_node_list': validNodeIDs,
               'circuit_simulation': circuitSimulation,
               'circuit_truthtable': truthTable,
               'circuit_response_functions': responseFunctions,
               'circuit_response_functions_particles': responseFunctionParameters,
               'circuit_envelop_bounding_mass': envelopBoundingMass,
               'circuit_original_input_output': originalInputOutput
               }

    simSpec = {'CPU_COUNT': multiprocessing.cpu_count(),
               'max_number_of_particles': maxNumberOfParticles}

    return simData, simSpec


"""
The method performing the actual simulation of the circuit depending on the provided assignment.

assignment: The precise assignment to simulate on the circuit of interest
simData: The data prepared during the initialisation phase, including response function parameters and the truthtables.
simSpec: Additional information currently not required 
"""


def startSimulation(assignment, simData, simSpec):
    # This method inserts the natively supported transfer functions for the later use during simulation.
    # TODO Remove this method from startSimulation and migrate to Lambda Expressions
    def insertNativeResponseFunctions(responseFunctions):
        keyList = responseFunctions.keys()
        for key in keyList:
            if (responseFunctions[key]["native"]):
                if (responseFunctions[key]["type"] == "INHIBITORY_HILL_EQUATION"):
                    responseFunctions[key]["equation"] = inhibitoryHillEquation
                elif (responseFunctions[key]["type"] == "IMPLICIT_OR"):
                    responseFunctions[key]["equation"] = nativeImplicitOr
                elif (responseFunctions[key]["type"] == "ENVELOPE"):
                    i = 0  # DO Nothing
            # The case of not native response functions is currently not treated yet, but is intented already by the design of the genetic gate library

        return responseFunctions

    # Creates a copy of the provided truthtable and replaces the values depending on the input specification whith the new ones.
    def updateTruthTable(truthTable, newInputSpecification, envelopMode):
        updatedTruthTable = truthTable.copy()
        inputIDs = updatedTruthTable["input_IDs"]
        updatedTruthTable["bio_inputs"] = updatedTruthTable[
            "bio_inputs"].copy()  # Copy to prevent manipulation of original truthtable

        updatedTruthTable["input_specification"] = newInputSpecification

        # Overwrite the input specification of the copied truthtable dict with the new values
        for inputIdentString in updatedTruthTable["inputs"]:
            combination = "".join([str(val) for val in updatedTruthTable["inputs"][inputIdentString]])
            updatedTruthTable["bio_inputs"][inputIdentString] = getBioVals(inputIDs, combination, newInputSpecification,
                                                                           envelopMode)
        return updatedTruthTable

    # Creates an inputspecification
    def generateInputSpecification(inputIDs, inLow=0.01, inHigh=3):
        inputSpecification = {inID: {"0": inLow, "1": inHigh} for inID in inputIDs}
        return inputSpecification

    """
    This method is the entrypoint for the actual simulation of the genetic circuit.
    In detail, this method iterates over the single particles, if they exist, and initiates the per particle/parameter simulation
    
    nodeOrder: The order in which the nodes shall be evaluated
    circuit: The circuit representation
    assignment: The assignment to simulate
    truthTable: The truthtable dict including input assignements and boolean outputs
    responseFunctions: The responsefunctions used within simulation
    particles: The particles to consider or not
    circuitSim: A dict acting as template for storing the circuit values or as direct store location
    """

    def startSimulation(nodeOrder, circuit, assignment, truthTable, responseFunctions, particles, circuitSim):
        # Restores the median transfer characteristics in case no particles are used.
        def restoreRealisations(realisations, responseFunctions):

            for realisation in realisations:
                debugPrint(
                    "Before restoring: (" + realisation + "): " + str(responseFunctions[realisation]["parameters"]))
                responseFunctions[realisation]["parameters"] = dict.copy(
                    responseFunctions[realisation]["library_parameters"])
                debugPrint(
                    "After restoring: (" + realisation + "): " + str(responseFunctions[realisation]["parameters"]))

            return realisations

        numberOfParticlesToUse = 0
        usedRealisations = assignment.values()

        # Differentiate whether particles shall be used or not
        if (simContext["particles"] == True):
            numberOfParticlesToUse = min([particles["max_number_of_particles"],
                                          simContext["numberOfParticles"]])
            usermodePrint("Used particles:" + str(numberOfParticlesToUse))
            # Restoring is not necessary, since the transfer characteristics parameters are overwritten subsequently
        else:
            numberOfParticlesToUse = 1
            restoreRealisations(usedRealisations, responseFunctions)
            usermodePrint("No particles are used")

        # Prepare the results dict
        results = {}
        for outID in truthTable["output_IDs"]:
            results[outID] = {"0": {}, "1": {}}

            for out in truthTable["outputs"]:
                # print(out)
                results[outID][truthTable["outputs"][out][outID]][out] = np.zeros(numberOfParticlesToUse)

        # Prepare storing the circuitvals in case the simulation should be visualised
        completeCircuitVals = {}
        if (simContext["visualise_circuit"]):
            completeCircuitVals = {key: {node: np.zeros(numberOfParticlesToUse) for node in nodeOrder} for key in
                                   truthTable["inputs"]}

        # The function used for simulation depends on whether substitution is performed or not.
        simulatorFunction = simulateCircuit
        if (simContext["substitute"]):
            simulatorFunction = simulateSubstitutedCircuit

        # Iterates over every particle for each parameter, while the circuit is simulated for each such particle
        for iX in range(numberOfParticlesToUse):

            # Insert particles into parameters

            # Check if particles shall be used
            if (simContext["particles"] == True):
                for node in usedRealisations:
                    # Insert particles into Response Function
                    try:  # Proof, whether the try catch can be replaced with if("particle_parameters" in respFunct):
                        respFunc = responseFunctions[node]
                        for parameter in respFunc["particle_parameters"]:
                            # print( particles[assignment[node]][parameter])
                            responseFunctions[node]["parameters"][parameter] = particles[node][parameter][iX]
                    except KeyError:
                        zero = 0

            # print(responseFunctions["NOT_0"])

            # Simulate Circuit
            # Either calls simulateCircuit or simulateSubstitutedCircuit (Depends on former if)
            simulatorFunction(nodeOrder, circuit, assignment, truthTable, responseFunctions, circuitSim, results,
                              completeCircuitVals, iX)

        return results, completeCircuitVals

    """
    !!!
    This method is only required for currently ongoing work and not used in case of a normal simulation.
    !!!
    The substitution is implemented on this level, since it is mathematically correct for single values.
    It has not been proven for particle simulation yet. 
    """

    def simulateSubstitutedCircuit(nodeOrder, circuit, assignment, truthTable, responseFunctions, circuitSim, results,
                                   completeCircuitVals, iteration):

        def determineInitialValuesToSubstitute(substitutionTruthTableKeys, responseFunctions, assignment, truthTable):
            inputIDs = truthTable["input_IDs"].keys()
            outputIDs = truthTable["output_IDs"].keys()
            valuesToSubstitute = {}

            for node in substitutionTruthTableKeys:
                if (node in inputIDs):
                    # Get the maximum value of this input
                    valueToSubstitute = truthTable["input_specification"][node]["1"]

                elif (not node in outputIDs):
                    bioGate = assignment[node]
                    valueToSubstitute = responseFunctions[bioGate]["parameters"]["ymax"]

                valuesToSubstitute[node] = valueToSubstitute

            return valuesToSubstitute

        def refineValuesToSubstitute(substitutionTruthtableKeys, substitutionTruthTables, currentCircuitVals,
                                     inputIdents, previousValuesToSubstitute):
            valuesToSubstitute = {}
            for key in substitutionTruthtableKeys:
                truthTable = substitutionTruthTables[key]
                maxVal = -np.infty
                valueSet = False
                for i in range(len(currentCircuitVals)):
                    if (truthTable[i] == "0"):
                        maxVal = np.max((currentCircuitVals[inputIdents[i]][key][0], maxVal))
                        valueSet = True

                # The case, when truthtable is 1 for every input combination.
                # Can be achieved only artificially, but is used by BAB Util to prevent iterative improvement
                # of artificially added inputs for which no evidence exists
                if (not valueSet):
                    maxVal = previousValuesToSubstitute[key]

                valuesToSubstitute[key] = maxVal

            return valuesToSubstitute

        substitutionTruthtables = truthTable['substitution_truthtables']
        substitutionTruthtableKeys = substitutionTruthtables.keys()

        inputIdents = list(truthTable["inputs"].keys())

        previousValuesToSubstitute = None;
        valuesToSubstitute = determineInitialValuesToSubstitute(substitutionTruthTableKeys=substitutionTruthtableKeys,
                                                                responseFunctions=responseFunctions,
                                                                assignment=assignment, truthTable=truthTable)

        i = 0
        stop = False
        while (not stop):
            debugPrint(str(valuesToSubstitute))

            # Clear current circuit vals
            currentCircuitVals = {key: {node: np.zeros(1) for node in nodeOrder} for key in
                                  truthTable["inputs"]}

            # Simulate based on the current valuesToSubstitute
            simulateCircuit(nodeOrder=nodeOrder, circuit=circuit, assignment=assignment, truthTable=truthTable,
                            responseFunctions=responseFunctions, circuitSim=circuitSim, results=results,
                            completeCircuitVals=currentCircuitVals, iteration=iteration,
                            substitutionValues=valuesToSubstitute)

            previousValuesToSubstitute = valuesToSubstitute
            valuesToSubstitute = refineValuesToSubstitute(substitutionTruthtableKeys=substitutionTruthtableKeys,
                                                          substitutionTruthTables=substitutionTruthtables,
                                                          currentCircuitVals=currentCircuitVals,
                                                          inputIdents=inputIdents,
                                                          previousValuesToSubstitute=previousValuesToSubstitute)

            stop = True
            for key in substitutionTruthtableKeys:
                stop = stop & (valuesToSubstitute[key] == previousValuesToSubstitute[key])
                if (not stop):
                    break
            # Stop when there is no change in the values to substitute, since the result would be identical

            determineScores(results)
            if (simContext["debug"]):
                for key in results:
                    debugPrint(str(key) + ": " + str(results[key]["SCORE"]))
            debugPrint("Finished iteration %d (stop=%s) \n" % (i, str(stop)))
            i += 1

        # The freshest result should be in result automatically

    """
    For a given set of parameters, this method adds the circuit outputs to the corresponding sets.
    The corresponding ID and either to the True or False set.
    
    
    nodeOrder: The order in which the nodes shall be evaluated
    circuit: The circuit representation
    assignment: The assignment to simulate
    truthTable: The truthtable dict including input assignements and boolean outputs
    responseFunctions: The responsefunctions used within simulation    
    circuitSim: A dict acting as template for storing the circuit values or as direct store location
    results: The map including the values for each circuit output. The current simulations results are added to this dict.
    completeCircuitVals: A field used to store every value present within the genetic circuit 
    iteration: The current iteration, respectively the index of the particle used for a given parameter
    substitutionValues: Values to use during substitution. (Only present if called by simulateSubstitutedCircuit)
    """

    def simulateCircuit(nodeOrder, circuit, assignment, truthTable, responseFunctions, circuitSim, results,
                        completeCircuitVals, iteration, substitutionValues={}):
        outputIDs = truthTable["output_IDs"].keys()

        # print(nodeOrder)
        bioInputs = truthTable["bio_inputs"]

        visualise = simContext["visualise_circuit"]

        substitute = simContext["substitute"]
        if (substitute):
            substitutions = truthTable["substitutions"]
            # valuesToSubstitute = substitutionValues
        else:
            substitutions = {inputIdent: {} for inputIdent in bioInputs}
            # valuesToSubstitute = {}

        saveCircuitVals = visualise or substitute
        # Iterates over the given inputs

        # Iterate over the output ids in order to simulate every input assignment
        for outputIdent in truthTable["outputs"]:
            # print(bioInputs["input_%d" % iX], truthTable["outputs"]["output_%d" % iX])
            inputIdent = outputIdent.replace("output",
                                             "input")  # Computes the inputidentfier for the current outputidentifier

            # Simulates the circuit for a particular input assignment
            circuitVals = getCircuitVal(nodeOrder, circuit, assignment, truthTable, bioInputs[inputIdent],
                                        dict(circuitSim), responseFunctions, substitution=substitutions[inputIdent],
                                        substitutionValues=substitutionValues)
            """
            print(iX, "\n", responseFunctions["NOT_0"]["parameters"], "\n", circuitVals, "\n")
            """

            debugPrint("circuitVals: (" + inputIdent + ") " + str(circuitVals))
            # in case the values should be saved,
            if (saveCircuitVals):  # Is true if already one of visualise circuit and substitute circuit is true
                for nodeIdent in nodeOrder:
                    completeCircuitVals[inputIdent][nodeIdent][iteration] = circuitVals[nodeIdent]
                # completeCircuitVals[inputIdent] = dict(circuitVals)

            # Store the circuit outputs in the result list
            for outputID in outputIDs:
                results[outputID][truthTable["outputs"][outputIdent][outputID]][outputIdent][iteration] = circuitVals[
                    outputID]

    """
    Evaluates a complete circuit according to the specified nodeOrder
    The parameters provided by the responseFunctions are used.
    
    nodeOrder: The order in which the nodes shall be evaluated
    circuit: The circuit representation
    assignment: The assignment to simulate
    truthTable: The truthtable dict including input assignements and boolean outputs
    circuitInputs: The concentrations representing the circuit input assignment
    circuitVals: The dict including the values within the circuit
    responseFunctions: The responsefunctions used within simulation    
    substitution: The map giving rise to which values need to be substituted (Only present if called by simulateSubstitutedCircuit)
    substitutionValues: Values to use during substitution. (Only present if called by simulateSubstitutedCircuit)
    """

    def getCircuitVal(nodeOrder, circuit, assignment, truthTable, circuitInputs, circuitVals, responseFunctions,
                      substitution={}, substitutionValues={}):
        inputIDs = truthTable["input_IDs"].keys()
        outputIDs = truthTable["output_IDs"].keys()
        # Iterate over the nodes in the circuit according to the order specified by nodeOrder
        for node in nodeOrder:
            # Case distinction: The nodes need to be treated differently, according to whether they are Inputs, Logic Gates or the outputs
            if (node in inputIDs):
                # The value of the current node is given by the circuit inputs
                value = circuitInputs[node]  # *np.exp(np.random.normal(scale=simContext["log_input_std"]))
            elif (node in outputIDs):
                # Theoretically we are done and can return
                # A final operation to the output value can be applied. This is currently not the case.
                value = circuitVals[node]
            else:
                # Apply the corresponding transfer characteristic to the gates input values
                bioGate = assignment[node]
                value = evaluateGate(circuitVals[node], responseFunctions[bioGate])

            circuitVals[node] = value

            # Check if substitution occurs
            nodeInSubstitution = node in substitution
            if (nodeInSubstitution):
                valueToSubstitute = substitutionValues[node]

            # Propagate the gates output value to each of its subsequent gates
            for child in circuit[node]:
                valueToPropagate = value if (not (nodeInSubstitution and child in substitution[node])) \
                    else valueToSubstitute  # The actual value propagated depends on whether the target gate requires a substitued gate or not.

                circuitVals[child] += valueToPropagate
                # print(node, circuitVals[node])
        # print("OUTPUT", circuitVals["NOT_1"])
        # print("")
        return circuitVals

    """
    Evaluates a gate for the given response function and the corresponding parameters
    """

    def evaluateGate(val, responseFunction):
        if (responseFunction["native"]):
            return responseFunction["equation"](val, responseFunction["parameters"])
        else:
            return -1  # Not implemented yet...

    """
        Implement the natively supported response functions
    """

    def inhibitoryHillEquation(x, parameters):
        # print(parameters)
        K = parameters["K"]
        n = parameters["n"]
        ymin = parameters["ymin"]
        ymax = parameters["ymax"]
        return ymin + (ymax - ymin) / (1 + (x / K) ** n)

    def nativeImplicitOr(x, parameters):
        return x;

    """
    The scoring functions
    """

    # Creates every possible ON and OFF Combination of the output and uses the minimum as the score. For the actual scoring, the scoring function is used.
    def determineScores(simRes):
        minON = {}
        maxOFF = {}
        # Determine the score for each Output individually
        for key in simRes:
            simRes[key]["SCORE"] = -1
            # print("Len ons:", len(simRes[key]["1"]))
            # print("Len offs:", len(simRes[key]["0"]))
            debugPrint("Output: " + key)
            debugPrint("ON: " + str(simRes[key]["1"]))
            debugPrint("OFF: " + str(simRes[key]["0"]))

            # Determine the minimum of the scoring function
            for outON in simRes[key]["1"]:
                for outOFF in simRes[key]["0"]:

                    # Important for the cello metric, that on is data 1 and off is data 2
                    score = circuitScore(simRes[key]["1"][outON], simRes[key]["0"][outOFF])

                    if (simRes[key]["SCORE"] == -1 or simRes[key][
                        "SCORE"] > score):  # The current value is overwritten in case a new minimum has been obtained
                        simRes[key]["SCORE"] = score
                        minON[key] = simRes[key]["1"][outON]
                        maxOFF[key] = simRes[key]["0"][outOFF]

        if (simContext["visualise"]):
            # saveSimResults(simRes)
            for key in minON:
                print("Smallest Distance for", key)
                logHistWrapperSubplot(minON[key], maxOFF[key])

    # The scoring for the envelop based circuit.
    # Since the equivalent envelop free circuit consists of non independent circuit outputs,
    # a different treatment is necessary
    def determineEnvelopBasedScores(simRes, lowerBoundMass=None):
        # A simple scoring approach yielding an interval which includes the actual score.
        def boundCalculation(simRes):
            intermediateSimRes = {"O_L": {},
                                  "O_H": {}}
            # Scheme:  [score(minOn_low, maxOff_high), score(minOn_high, maxOff_low)]
            # TODO implement the ability of scoring other output names as "O"
            intermediateSimRes["O_L"]["1"] = simRes["O_L"]["1"]
            intermediateSimRes["O_L"]["0"] = simRes["O_H"]["0"]

            intermediateSimRes["O_H"]["1"] = simRes["O_H"]["1"]
            intermediateSimRes["O_H"]["0"] = simRes["O_L"]["0"]
            determineScores(intermediateSimRes)

            return intermediateSimRes

        # Reshaping of simulation results for the later scoring
        intermediateSimRes = {key[:-2]: {"0": {},
                                         "1": {}}
                              for key in
                              simRes.keys()}  # The identifier of gates representing upper and lower envelop are removed to combine these values

        for key in intermediateSimRes:
            lowerIdent = key + "_L"
            upperIdent = key + "_H"
            intermediateSimRes[key]["0"] = {key: [] for key in simRes[lowerIdent]["0"].keys()}
            intermediateSimRes[key]["1"] = {key: [] for key in simRes[lowerIdent]["1"].keys()}

            levels = intermediateSimRes[key].keys()  # Level refers to either "0" or "1"
            for level in levels:
                outputIdents = intermediateSimRes[key][level].keys()
                for outputIdent in outputIdents:
                    # The values of the lower envelope are included in the first list and the one of the upper envelopes in the second
                    intermediateSimRes[key][level][outputIdent].append(simRes[lowerIdent][level][outputIdent])
                    intermediateSimRes[key][level][outputIdent].append(simRes[upperIdent][level][outputIdent])
                    intermediateSimRes[key][level][outputIdent] = np.array(intermediateSimRes[key][level][outputIdent])

        dist = simContext["dist"]
        # Again, the scores are determined for each output separately
        for output in intermediateSimRes:
            currentScore = None
            currentOFF_CDF = None
            currentON_CDF = None

            if (dist == "ev-bound"):  # Determines the lower bound of the interval including the actual score
                OFF_Data = intermediateSimRes[output]['0']
                ON_Data = intermediateSimRes[output]['1']
                OFF_Data = [OFF_Data[elem][1] for elem in OFF_Data]  # [1] Selects the Upper Envelope
                ON_Data = [ON_Data[elem][0] for elem in ON_Data]  # [0] Selects the Lower Envelope
                lowestOnL = np.min(ON_Data)  # The lowest value corresponding to an Lower envelope is taken into account
                highestOffH = np.max(
                    OFF_Data)  # The highest value corresponding to an upper envelope is taken into account

                score = lowestOnL / highestOffH  # Simply calculate the cello score

                currentScore = score

            else:
                # Get the probability masses the envelopes bound for the current output
                currentLowerMassBounds = [entry[output] for entry in lowerBoundMass]

                # Determine the piecewise linear CDF representing the distribution of the OFF concentrations based on the envelopes
                intermediateSimRes[output]["0_CDF"] = {}
                OFF_CDFs = []
                for OFF_ident in intermediateSimRes[output][
                    "0"]:  # For each input assignment, a dedicated output CDF is determined
                    OFF_Bounds = intermediateSimRes[output]["0"][OFF_ident]
                    CDF = see.determineCDFNew(OFF_Bounds, currentLowerMassBounds)
                    OFF_CDFs.append(CDF)  # Collect the CDFs corresponding to the same boolean state in a list
                    intermediateSimRes[output]["0_CDF"][OFF_ident] = CDF

                # Determine the piecewise linear CDF representing the distribution of the ON concentrations based on the envelopes
                intermediateSimRes[output]["1_CDF"] = {}
                ON_CDFs = []
                for ON_ident in intermediateSimRes[output][
                    "1"]:  # For each input assignment, a dedicated output CDF is determined
                    ON_Bounds = intermediateSimRes[output]["1"][ON_ident]
                    CDF = see.determineCDFNew(ON_Bounds, currentLowerMassBounds)
                    ON_CDFs.append(CDF)  # Collect the CDFs corresponding to the same boolean state in a list
                    intermediateSimRes[output]["1_CDF"][ON_ident] = CDF

                # Again, determine the minimum distance between the CDFs over all possible combinations
                for OFF_CDF in OFF_CDFs:
                    for ON_CDF in ON_CDFs:
                        score = see.scoreEnvelopOutput(OFF_CDF=OFF_CDF, ON_CDF=ON_CDF, dist=dist)

                        if (
                                currentScore == None or score < currentScore):  # If a new minimal score has been observed, use this as score
                            currentScore = score
                            currentOFF_CDF = OFF_CDF
                            currentON_CDF = ON_CDF

                if (simContext["visualise"]):
                    # The scoring method also has the ability of visualising the scored CDFs
                    see.scoreEnvelopOutput(OFF_CDF=currentOFF_CDF, ON_CDF=currentON_CDF, dist=dist, visualise=True,
                                           PLOT_CDFs=sve.envelope_plotCDFs, PLOT_CDFs_DIFF=sve.envelope_plotCDFsDiff)

            intermediateSimRes[output][
                "SCORE"] = currentScore  # Set the obtained score as score for the considered output
        return intermediateSimRes

    envelopMode = simContext["envelop_mode"]

    # Extract the relevant information from simData
    nodeOrder = simData["circuit_valid_node_list"]
    circuit = simData["circuit"]
    originalCircuit = simData["circuit_original"]
    truthTable = simData["circuit_truthtable"]
    circuitVals = simData["circuit_simulation"]
    responseFunctions = simData["circuit_response_functions"]
    particles = simData["circuit_response_functions_particles"]

    envelopBoundingMass = simData["circuit_envelop_bounding_mass"]
    originalInputOutput = simData["circuit_original_input_output"]

    # The supported transfer characteristics are referenced
    insertNativeResponseFunctions(responseFunctions)

    # The assignment also needs to be transformed in order to use it within the equivalent envelop free circuit
    if (envelopMode):
        nToUse = min([particles["max_number_of_particles"], simContext["numberOfParticles"]]) if (
            simContext["particles"]) else 1

        lowerBoundMass = None
        if (simContext["dist"] != "ev-bound"):
            # The probability mass each envelope bounds needs to be determined for the later scoring.
            # This is not necessary for "ev-bound", since this score only takes the pure intervals into account
            lowerBoundMass = see.determineProbabilityMass(originalCircuit, envelopBoundingMass,
                                                          originalInputOutput=originalInputOutput,
                                                          originalAssignment=assignment, n=nToUse)
        assignment = see.parseAssignmentToEnvelopeAssignment(assignment)

    # Set the truthtable to use during simulation
    # The transfer in an extra variable is necessary, since this truthtable can be overwritten
    truthTableToUse = truthTable
    # In case a custom input specification per simulation is defined, this is used to create an updated truthtable
    if (simContext["use_custom_input_specification"]
            and (("custom_input_low" in simContext
                  and "custom_input_high" in simContext) or simContext["custom_input_specification"] != None)):

        inputSpecification = simContext["custom_input_specification"]
        if (inputSpecification == None):
            # Fallback mechanism in case no custom input specification has been provided but only the general low and high concentrations
            inputSpecification = generateInputSpecification(truthTable["input_IDs"], simContext["custom_input_low"],
                                                            simContext["custom_input_high"])
        updatedTruthTable = updateTruthTable(truthTable=truthTable, newInputSpecification=inputSpecification,
                                             envelopMode=envelopMode)

        # Set the truthtable to use during simulation
        truthTableToUse = updatedTruthTable
        # Reset in order to enable explicit overwrite during simulation and automatic reconstruction of the initialisation setting if cis=0
        simContext["use_custom_input_specification"] = False

    """
        Perform Simulation.
    """

    simRes, completeCircuitVals = startSimulation(nodeOrder, circuit, assignment, truthTableToUse, responseFunctions,
                                                  particles,
                                                  dict(circuitVals))

    # debugPrint("simRes: \n" + str(simRes))
    """
        Determine Scores of all output variables
    """

    # The scoring method used depends on whether envelope mode is used or not
    if (envelopMode):
        simRes = determineEnvelopBasedScores(simRes, lowerBoundMass)
    else:
        determineScores(simRes)

    # Stores the simulation data
    if (simContext["store"] == True):
        saveSimResults(truthTable, simRes, completeCircuitVals=completeCircuitVals)

    usermodePrint("Circuit Scores: ")
    # print("Circuit Score:", simRes["O"]["SCORE"])
    # for outputID in truthTable["output_IDs"].keys():
    for outputID in simRes:
        # print(outputID, simRes[outputID]["SCORE"])
        # TODO: Remove this. It shouldn't be suppressed here, but enuMap crashes with YFP output
        if outputID != 'YFP':
            print(outputID, simRes[outputID]["SCORE"])

    # Methods for visualising the values within the circuit
    if (simContext["visualise_circuit"]):
        if (simContext["particles"] == True):
            sve.plotCompleteCircuitValsHistogramms(completeCircuitVals)
        else:
            sve.visualiseCircuitWithValues(circuit=circuit, assignment=assignment, responseFunctions=responseFunctions,
                                           circuitValues=completeCircuitVals)

        # print("Visualise Circuit does only work partly in this version. The data is collected but not plotted.")


##############################################################################

"""
    Is used for each simulation as well as for the initialisation.
"""

# This method parses the command line string
def parseInput(inputText):
    # This method ensures, that the string is in the right format
    def prepareString(prepText):
        prepText = " ".join(prepText.split())
        prepText = prepText.replace(" =", "=")
        prepText = prepText.replace("= ", "=")
        prepText = prepText.replace(" :", ":")
        prepText = prepText.replace(": ", ":")
        prepText = prepText.replace(" ,", ",")
        prepText = prepText.replace(", ", ",")
        return prepText

    # A method aiding in parsing the different values applicable to represent a boolean state
    def parseBool(inVal):
        curVal = inVal.lower();
        if (curVal == "0" or curVal == "false" or curVal == "f"):
            curVal = 0;
        elif (curVal == "1" or curVal == "true" or curVal == "t"):
            curVal = 1;
        return curVal

    inputText = prepareString(inputText)
    instruction = inputText.split()  # Split at Whitespace!

    if (len(instruction) > 0):
        cmd = instruction[0]    # Format: command [optional args]
        specDict = {}
        # Parse the arguments for the given command. No differentiation which command is present
        for elem in instruction[1:]:
            field, val = elem.split("=")

            if (field == "n"):
                specDict["numberOfParticles"] = int(val)
            elif (field == "particles" or field == "p"):
                specDict["particles"] = bool(parseBool(val))
            elif (field == "nMax"):
                specDict["maxNumberOfParticles"] = int(val)
            elif (field == "threads"):
                specDict[field] = int(val)
            elif (field == "lib_path"):
                specDict["gate_lib"] = loadJSON(val)
            elif (field == "s_path"):
                specDict["structure"] = loadJSON(val)
            elif (field == "structure"):
                specDict["structure"] = ast.literal_eval(val)
            elif (field == "a_path"):
                specDict["assignment"] = loadJSON(val)
            elif (field == "assignment"):
                specDict["assignment"] = ast.literal_eval(val)
            elif (field == "visualise"):
                specDict[field] = bool(parseBool(val))
            elif (field == "visualise_circuit" or field == "vc"):
                specDict["visualise_circuit"] = bool(parseBool(val))
            elif (field == "whitelist" or field == "wh"):
                specDict["whitelist"] = bool(parseBool(val))
            elif (field == "substitute"):
                specDict["substitute"] = bool(parseBool(val))
            elif (field == "store"):
                specDict[field] = bool(parseBool(val))
            elif (field == "usermode"):
                specDict[field] = bool(parseBool(val))
            elif (field == "debug"):
                specDict[field] = bool(parseBool(val))
            elif (field == "dist"):
                specDict[field] = val.lower()
            elif (field == "envelope_mode" or field == "envelop_mode" or field == "em"):
                specDict["envelop_mode"] = bool(parseBool(val))
            elif (field == "particle_quantiles" or field == "pq"):
                specDict["particle_quantiles"] = ast.literal_eval(val)
            elif (field == "use_custom_input_specification" or field == "ucis"):
                specDict["use_custom_input_specification"] = bool(parseBool(val))
            elif (field == "custom_input_specification" or field == "cis"):
                specDict["custom_input_specification"] = ast.literal_eval(val)
            elif (field == "custom_input_low" or field == "ciL"):
                specDict["custom_input_low"] = float(val)
            elif (field == "custom_input_high" or field == "ciH"):
                specDict["custom_input_high"] = float(val)
            elif (field == "wasserstein_p"):
                specDict["wasserstein_p"] = float(val)
            elif (field == "log_input_std"):
                specDict["log_input_std"] = float(val)
            elif (field == "store_suffix"):
                specDict[field] = val.lower().replace('"', '').replace("'", "")
            else:
                specDict[field] = val
        return cmd, specDict
    else:
        return "", {}

# A helper method which can be used to print information relevant for any user
def usermodePrint(text):
    if (simContext["usermode"]):
        print(text, file=sys.stdout)

# A helper method for printing debug information in debug mode
def debugPrint(text):
    if (simContext["debug"]):
        print(text, file=sys.stderr)

# A method updating the simulation context based on the command args
def updateSimContext(specDict):
    for key in specDict:
        if (key in simContext):
            simContext[key] = specDict[key]


print("Start Simulator Initialisation")

# Default Values for the simulator Settings
simContext = {}
simContext["particles"] = False;  # Can be changed at each simulation since original parameters are restored
simContext["numberOfParticles"] = 5000;  # Changeable at each simulation
simContext["maxNumberOfParticles"] = 5000;  # Can only be set at the initialisation of the simulator
simContext["visualise"] = False;
simContext["visualise_circuit"] = False
simContext["whitelist"] = False  # Can only be set at the initialisation of the simulator
simContext["substitute"] = False
simContext["store"] = False;
simContext["usermode"] = False;
simContext["debug"] = False;
simContext["structure"] = "NULL";  # Can only be set at the initialisation of the simulator
simContext["assignment"] = "NULL";
simContext["gate_lib"] = "NULL";  # Can only be set at the initialisation of the simulator
simContext["threads"] = 1;  # Currently not used
simContext["dist"] = "ws-log-exp-asym";  # The recommended scoring method
simContext["envelop_mode"] = False  # Can only be set at the initialisation of the simulator (if dist not set explicitly then ev-asym is used)
simContext["particle_quantiles"] = None
simContext["use_custom_input_specification"] = False  # Can only be set at the initialisation of the simulator
simContext["custom_input_specification"] = None
simContext["custom_input_low"] = 0.00001
simContext["custom_input_high"] = 10
simContext["wasserstein_p"] = 2.0
simContext["log_input_std"] = 0.  # have a clean input bei default
simContext["store_suffix"] = str(int(time.time()))

debugPrint(str(sys.argv))

# Parse the initial command
cmd, specDict = parseInput(" ".join(sys.argv))
updateSimContext(specDict)

# Set the maximum number of particles as default
simContext["numberOfParticles"] = simContext["maxNumberOfParticles"];

# Check if initialisation was successfull.
if (simContext["structure"] != "NULL" and simContext["gate_lib"] != "NULL"):
    # Continue here, if a structure and a gate lib have been passed
    simData, simSpec = initialiseSimulator(simContext["structure"], simContext["gate_lib"],
                                           simContext["maxNumberOfParticles"])
    usermodePrint("Simulator Initialised and ready to use.")
    usermodePrint(
        "Dont forget to configure the assignment (append \"start\" with \"a_path=...\", the corresponding path, or directly by the assignment, by making use of\"assignment=...\"=  and the number of particles (append \"n=...\" to \"start\")")
    usermodePrint("Start simulation with \"start\"")
else:
    # Continue here if the passed arguments are not sufficient
    usermodePrint("You need to set the circuit structure and the library path to initialise the simulation!")
    usermodePrint(
        "This can be done by appending the command \"settings\" by \"s_path=...\" and the corresponding file path and next to this, the path to the gate lib with \"lib_path=...\"")
    structurePath = "NULL"
    gateLibPath = "NULL"
    configured = False
    inText = input("define settings:\n")
    # This loop ensures, that the basic command loop is only entered with a valid configuration
    while (inText != "exit" and not (configured)):
        cmd, specDict = parseInput(inText)
        debugPrint(cmd)
        if (cmd == "settings"):
            updateSimContext(specDict)

            if (simContext["structure"] != "NULL" and simContext["gate_lib"] != "NULL"):
                simData, simSpec = initialiseSimulator(simContext["structure"], simContext["gate_lib"],
                                                       simContext["maxNumberOfParticles"])

                usermodePrint("Configuration finished")
                usermodePrint("If desired, this configuration can be changed later on with the same comands.")
                configured = True

        if (not (configured)):
            inText = input("define settings:\n")

    if (inText == "exit"):
        sys.exit("You stopped the simulator.")

inText = input("ready:\n")

specDict = {}
simSpec["max_number_of_particles"] = simContext["maxNumberOfParticles"]
iX = 0
# The basic command handling loop
while (inText != "exit"):
    cmd, specDict = parseInput(inText)
    # Currently the three commands "exit", "start" and "settings" are defined
    # "exit" terminates the execution of the simulator
    # "start" starts the simulation of a assignment. One of a_path or assigment must be defined at least for the first simulation.
    # The subsequent simulations make use of the same assignment if not specified otherwise
    # "settings" reinitilizes the simulator with the new parameters comparable as if it was started again

    if (cmd == "start"):
        updateSimContext(specDict)

        if (simContext["assignment"] != "NULL"):
            # print(assignment)
            usermodePrint("Simulation started")
            startSimulation(assignment=simContext["assignment"], simData=simData, simSpec=simSpec) # Perform simulation
            debugPrint("Iteration: " + str(iX))
            iX += 1
        else:
            print("ERROR: Assignment needs to be defined at least once by adding \"assignment={...}\" or \"a_path=dir/file.json\" to after start.")

    if (cmd == "settings"):
        updateSimContext(specDict)

        simData, simSpec = initialiseSimulator(simContext["structure"], simContext["gate_lib"],
                                               simContext["maxNumberOfParticles"])  # Update data fields
        usermodePrint("Simulator is reinitialised")

    inText = input("")
