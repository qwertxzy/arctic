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
            i += 0.5    # Increment only by 0.5, since in envelopMode two inputs correspond to the same boolean state
        else:
            bioVals[inputID] = inputSpecification[inputID][str(combination[int(i)])]
            i += 1      # Increment by one, to get the next entries boolean value.

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
    sign = 1 if np.median(dataON) > np.median(dataOFF) else -1  # Determine the sign of the exponent based on the median differences
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
    This method currently only works for one parameter!
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

        particlesLib = {"max_number_of_particles": n}

        if (simContext["envelop_mode"]):
            see.prepareEnvelopeParameters(particlesLib, gateLib, n)
            return particlesLib

        for gate in gateLib:
            if ("particles" in gate["biorep"]):
                # Add placeholder for particles of Gate "gate"
                particlesLib[gate['identifier']] = {}

                # Obtain all parameters for which particles are provided or shall be provided
                for parameter in gate['biorep']['particles']:
                    rawParticles = gate['biorep']['particles'][parameter]

                    numberOfProvidedParticles = len(rawParticles)
                    if (numberOfProvidedParticles > 0):

                        # Generate additional particles, if the number of particles provided is smaller then the number of particles requested
                        if (numberOfProvidedParticles < n):
                            additionalParticles = generateParticles(np.mean(rawParticles),
                                                                    n - numberOfProvidedParticles)

                        if (numberOfProvidedParticles > n):
                            numberOfProvidedParticles = n

                        particles = np.zeros(n)

                        for iX in range(numberOfProvidedParticles):  # Add all provided particles
                            particles[iX] = rawParticles[iX]

                        for iX in range(numberOfProvidedParticles, n):  # Add the additionally generated particles
                            particles[iX] = additionalParticles[iX - numberOfProvidedParticles]

                    else:
                        particles = generateParticles(gate['biorep']['response_function']["parameters"][parameter], n)

                    # Assign the particles
                    particlesLib[gate['identifier']][parameter] = particles

        return particlesLib

    #   This method checks whether the circuit is represented in the right order,
    #   thus the gates are in the order corresponding to the calculation.
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
    """

    def createValidNodeIDList(nodes, edges):
        def allSourcesPresent(edges, nodeID, validNodeIDs):
            targetSources = getSources(edges, target)
            for targetSource in targetSources:
                if (not targetSource in validNodeIDs):
                    return False
            return True

        validNodeIDs = []

        nodeSets = getSets(nodes)
        sortedInputs = sorted(nodeSets['INPUT'], key=lambda d: d['id'])

        for inVal in sortedInputs:
            validNodeIDs.append(inVal['id'])

        i = 0
        while (len(validNodeIDs) != len(nodes)):
            targets = getTargets(edges, validNodeIDs[i])
            for target in targets:
                if (not target in validNodeIDs):
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

    def generateTruthTable(inputIDs, outputIDs, outputTruthTable, inputSpecification, envelopMode=False,
                           whiteListString=None, substitutionsList=None, substitutionTruthtables=None):

        sortedInputIDs = sorted(inputIDs)
        sortedOutputIDs = sorted(outputIDs)

        inputIDs = {sortedInputIDs[val]: val for val in range(len(sortedInputIDs))}
        outputIDs = {sortedOutputIDs[val]: val for val in range(len(sortedOutputIDs))}

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
        if (envelopMode):
            if (numberOfInputs % 2 != 0):
                raise Exception("Some problem in creation of inputIDs")
            numberOfInputs = int(numberOfInputs / 2)

        for i in range(len(outputTruthTable)):
            if (whiteListString != None and whiteListString[i] == "0"):
                continue
            combination = getCombination(i, numberOfInputs)
            inputIdentString = "input_" + str(i)
            truthTable["inputs"][inputIdentString] = [int(c) for c in combination]
            identString = "output_" + str(i)
            truthTable["outputs"][identString] = {}  # Only works for just one circuit output
            for outID in sortedOutputIDs:
                truthTable["outputs"][identString][outID] = outputTruthTable[i]

            truthTable["bio_inputs"][inputIdentString] = getBioVals(inputIDs, combination, inputSpecification,
                                                                    envelopMode)

            if (substitutionsList != None):
                truthTable["substitutions"][inputIdentString] = substitutionsList[i]
            else:
                truthTable["substitutions"][inputIdentString] = {}

            if (substitutionTruthtables != None):
                for key in substitutionTruthtables:
                    truthTable["substitution_truthtables"][key] += substitutionTruthtables[key][i]

        return truthTable

    """
        This specification shall be given within the library or any additional source.
    """

    def generateInputSpecification(inputIDs, inLow=0.01, inHigh=3):
        inputSpecification = {inID: {"0": inLow, "1": inHigh} for inID in inputIDs}
        return inputSpecification

        # Only applicable for three inputs

    def generateCelloInputSpecification(inputIDs):
        inputSpecification = {inputID: {"0": 0.0082, "1": 2.5} for inputID in inputIDs}
        # Overwrites Cello specific input IDs with Cello specific values.
        inputSpecification["a"] = {"0": 0.0034, "1": 2.8}
        inputSpecification["b"] = {"0": 0.0013, "1": 4.4}
        inputSpecification["c"] = {"0": 0.0082, "1": 2.5}
        inputSpecification["d"] = {"0": 0.025, "1": 0.31}
        return inputSpecification

    # Only applicable for three inputs
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

    def generateEnvelopeInputSpecification(inputIDs):
        inputSpecification = {}
        for key in inputIDs:
            keyLen = len(key)
            if (key[keyLen - 2:] == "H"):
                inputSpecification[key] = {"0": 0.01, "1": 3}  # Upper Envelope Part
            else:
                inputSpecification[key] = {"0": 0.001, "1": 2}  # Lower Envelope Part

        return inputSpecification

    # jsonContent = loadJSON(structurePath)
    # jsonContent = structureFileContent
    # structureData = jsonContent['graph']
    # simContext["numberOfParticles"] = simContext["maxNumberOfParticles"];

    nodes = structureContent['graph']['nodes']
    edges = structureContent['graph']['edges']
    nodeDict = getNodeDict(nodes)
    nodeSets = getSets(nodes)

    validNodeIDs = createValidNodeIDList(nodes, edges)
    validNodes = sortNodes(validNodeIDs, nodeDict)

    inputIDs = [node['id'] for node in nodeSets['INPUT']]
    outputIDs = [node['id'] for node in nodeSets['OUTPUT']]

    envelopMode = simContext["envelop_mode"]

    circuit = generateCircuit(nodeDict, edges, validNodeIDs)

    originalCircuit = circuit.copy()

    if (envelopMode):
        envelopeCircuit, envelopeInputIDs, envelopeOutputIDs = see.generateEnvelopCircuit(circuit=circuit,
                                                                                          inputIDs=inputIDs,
                                                                                          outputIDs=outputIDs,
                                                                                          nodeDict=nodeDict)

    # TODO Adapt circuitSimulation, inputIDs and outputIDs to envelopMode
    circuitInfo = {'NUMBER_OF_INPUTS': len(nodeSets['INPUT']),
                   'NUMBER_OF_OUTPUTS': len(nodeSets['OUTPUT']),
                   'NUMBER_OF_LOGICS': len(nodeSets['LOGIC'])}

    if (simContext["use_custom_input_specification"]
            and (("custom_input_low" in simContext
                  and "custom_input_high" in simContext) or simContext["custom_input_specification"] != None)):

        inputSpecification = simContext["custom_input_specification"]
        if (inputSpecification == None):
            inputSpecification = generateInputSpecification(inputIDs if (not envelopMode) else envelopeInputIDs,
                                                            simContext["custom_input_low"],
                                                            simContext["custom_input_high"])
        simContext[
            "custom_input_specification"] = False  # Reset in order to enable explicit overwrite during simulation and automatic reconstruction of the initialisation setting if cis=0
    # elif (len(inputIDs) == 3):
    else:
        if (envelopMode):
            inputSpecification = generateCelloEnvelopeInputSpecification(envelopeInputIDs)
        else:
            inputSpecification = generateCelloInputSpecification(inputIDs)
    #    else:
    #        if (envelopMode):
    #            inputSpecification = generateEnvelopeInputSpecification(envelopeInputIDs)
    #        else:
    #            inputSpecification = generateInputSpecification(inputIDs)

    truthTableString = structureContent['truthtable']
    truthTableString = truthTableString[len(truthTableString)::-1]

    whiteListString = None
    if (simContext["whitelist"] and "whitelist" in structureContent):
        whiteListString = structureContent["whitelist"]
        whiteListString = whiteListString[len(whiteListString)::-1]

    substitutionslist = None
    if ('substitutions_list' in structureContent):
        substitutionslist = structureContent['substitutions_list']

    substitutionTruthtables = None
    if ('substitution_truthtables' in structureContent):
        substitutionTruthtables = structureContent['substitution_truthtables']
        for key in substitutionTruthtables:
            # Reverse the truthtables
            substitutionTruthtables[key] = substitutionTruthtables[key][len(substitutionTruthtables[key])::-1]

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

    validNodeIDs = list(circuit.keys())
    circuitSimulation = {nodeID: 0 for nodeID in validNodeIDs}
    # gateLib = loadJSON(gateLibPath)
    responseFunctions = generateResponseFunctions(gateLib)

    # Load particles if present or generate new (no differentiation between simulation with or without particles currently)
    # If not enough particles are given, additional ones will be generated according to the mean of provided particles
    responseFunctionParameters = prepareParameters(gateLib, maxNumberOfParticles)

    envelopBoundingMass = None
    if (envelopMode):
        envelopBoundingMass = see.prepareEnvelopBoundingMass(gateLib)

    originalInputOutput = {"input_IDs": inputIDs,
                           "output_IDs": outputIDs}

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


def startSimulation(assignment, simData, simSpec):
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
                    # TODO: Check, whether this needs a change for particle support of envelopes

        return responseFunctions

    # Envelop Mode is currently not supported!
    def updateTruthTable(truthTable, newInputSpecification, envelopMode):
        updatedTruthTable = truthTable.copy()
        inputIDs = updatedTruthTable["input_IDs"]
        updatedTruthTable["bio_inputs"] = updatedTruthTable[
            "bio_inputs"].copy()  # Copy to prevent manipulation of original truthtable

        updatedTruthTable["input_specification"] = newInputSpecification

        for inputIdentString in updatedTruthTable["inputs"]:
            combination = "".join([str(val) for val in updatedTruthTable["inputs"][inputIdentString]])
            updatedTruthTable["bio_inputs"][inputIdentString] = getBioVals(inputIDs, combination, newInputSpecification,
                                                                           envelopMode)
        return updatedTruthTable

    def generateInputSpecification(inputIDs, inLow=0.01, inHigh=3):
        inputSpecification = {inID: {"0": inLow, "1": inHigh} for inID in inputIDs}
        return inputSpecification

    def startSimulation(nodeOrder, circuit, assignment, truthTable, responseFunctions, particles, circuitSim):
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
        else:
            numberOfParticlesToUse = 1
            restoreRealisations(usedRealisations, responseFunctions)
            usermodePrint("No particles are used")

        results = {}
        for outID in truthTable["output_IDs"]:
            results[outID] = {"0": {}, "1": {}}

            for out in truthTable["outputs"]:
                # print(out)
                results[outID][truthTable["outputs"][out][outID]][out] = np.zeros(numberOfParticlesToUse)

        completeCircuitVals = {}
        if (simContext["visualise_circuit"]):
            completeCircuitVals = {key: {node: np.zeros(numberOfParticlesToUse) for node in nodeOrder} for key in
                                   truthTable["inputs"]}

        simulatorFunction = simulateCircuit
        if (simContext["substitute"]):
            simulatorFunction = simulateSubstitutedCircuit

        for iX in range(numberOfParticlesToUse):

            # Insert particles into parameters

            # Check if particles shall be used
            if (simContext["particles"] == True):
                for node in usedRealisations:
                    # Insert particles into Response Function            for node in usedRealisations:
                    try:  # Proof, whether the try catch can be replaced with if("particle_parameters" in respFunct):
                        respFunc = responseFunctions[node]
                        for parameter in respFunc["particle_parameters"]:
                            # print( particles[assignment[node]][parameter])
                            responseFunctions[node]["parameters"][parameter] = particles[node][parameter][iX]
                    except KeyError:
                        zero = 0

            # print(responseFunctions["NOT_0"])

            # Simulate Circuit
            # Either calls simulateCircuit or simulateSubstitutedCircuit
            simulatorFunction(nodeOrder, circuit, assignment, truthTable, responseFunctions, circuitSim, results,
                              completeCircuitVals, iX)

        return results, completeCircuitVals

    """
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

        # for iX in range(len(truthTable["outputs"])):
        for outputIdent in truthTable["outputs"]:
            # print(bioInputs["input_%d" % iX], truthTable["outputs"]["output_%d" % iX])
            inputIdent = outputIdent.replace("output", "input")
            # inputIdent = "input_%d" % iX
            circuitVals = getCircuitVal(nodeOrder, circuit, assignment, truthTable, bioInputs[inputIdent],
                                        dict(circuitSim), responseFunctions, substitution=substitutions[inputIdent],
                                        substitutionValues=substitutionValues)
            """
            print(iX, "\n", responseFunctions["NOT_0"]["parameters"], "\n", circuitVals, "\n")
            """

            debugPrint("circuitVals: (" + inputIdent + ") " + str(circuitVals))
            if (saveCircuitVals):  # Is true if already one of visualise circuit and substitute circuit is true
                for nodeIdent in nodeOrder:
                    completeCircuitVals[inputIdent][nodeIdent][iteration] = circuitVals[nodeIdent]
                # completeCircuitVals[inputIdent] = dict(circuitVals)

            # Store the circuit outputs in the result list
            for outputID in outputIDs:
                # results[outputID][truthTable["outputs"]["output_%d" % iX][outputID]]["output_%d" % iX][iteration] =  circuitVals[outputID]
                results[outputID][truthTable["outputs"][outputIdent][outputID]][outputIdent][iteration] = circuitVals[
                    outputID]
                # print("output_%d" % iX, truthTable["outputs"]["output_%d" % iX][outputID], results[outputID][truthTable["outputs"]["output_%d" % iX][outputID]]["output_%d" % iX][iteration])
                # results[outputID][truthTable["outputs"]["output_%d" % iX][outputID]]["output_%d" % iX][iX].append(circuitVals[outputID])

        # print("")

    """
    Evaluates a complete circuit according to the specified nodeOrder
    The parameters provided by the responseFunctions are used.
    """

    def getCircuitVal(nodeOrder, circuit, assignment, truthTable, circuitInputs, circuitVals, responseFunctions,
                      substitution={}, substitutionValues={}):
        inputIDs = truthTable["input_IDs"].keys()
        outputIDs = truthTable["output_IDs"].keys()
        for node in nodeOrder:
            if (node in inputIDs):
                # The value of the current node is given by the circuit inputs
                value = circuitInputs[node]  # *np.exp(np.random.normal(scale=simContext["log_input_std"]))
            elif (node in outputIDs):
                # Theoretically we are done and can return
                # A final operation to the output value can be applied. This is currently not the case.
                value = circuitVals[node]
            else:
                bioGate = assignment[node]
                value = evaluateGate(circuitVals[node], responseFunctions[bioGate])

            circuitVals[node] = value

            nodeInSubstitution = node in substitution
            if (nodeInSubstitution):
                # if (node in inputIDs):
                #     valueToSubstitute = truthTable["input_specification"][node][
                #         "1"]  # Get the maximum value of this input
                # elif (not node in outputIDs):
                #     valueToSubstitute = responseFunctions[bioGate]["parameters"]["ymax"]
                valueToSubstitute = substitutionValues[node]

            for child in circuit[node]:
                valueToPropagate = value if (not (nodeInSubstitution and child in substitution[node])) \
                    else valueToSubstitute

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

    def determineScores(simRes):
        minON = {}
        maxOFF = {}

        for key in simRes:
            simRes[key]["SCORE"] = -1
            # print("Len ons:", len(simRes[key]["1"]))
            # print("Len offs:", len(simRes[key]["0"]))
            debugPrint("Output: " + key)
            debugPrint("ON: " + str(simRes[key]["1"]))
            debugPrint("OFF: " + str(simRes[key]["0"]))
            for outON in simRes[key]["1"]:
                for outOFF in simRes[key]["0"]:

                    # if (np.median(simRes[key]["1"][outON]) > np.median(simRes[key]["0"][outOFF])):
                    # Important for the cello metric, that on is data 1 and off is data 2
                    score = circuitScore(simRes[key]["1"][outON], simRes[key]["0"][outOFF])

                    # else:
                    #    score = -2

                    # print(score);
                    # logHistWrapperSubplot(simRes[key]["1"][outON], simRes[key]["0"][outOFF])
                    if (simRes[key]["SCORE"] == -1 or simRes[key]["SCORE"] > score):
                        simRes[key]["SCORE"] = score
                        minON[key] = simRes[key]["1"][outON]
                        maxOFF[key] = simRes[key]["0"][outOFF]

        # print(visualise);
        # print(minON[key])
        # print(maxOFF[key])

        if (simContext["visualise"]):
            # saveSimResults(simRes)
            for key in minON:
                print("Smallest Distance for", key)
                logHistWrapperSubplot(minON[key], maxOFF[key])

    # Warning: Does not support multiple Envelopes, respectively envelopes based on multiple particles
    # Only the values of the first envelop are considered
    def determineEnvelopBasedScores(simRes, lowerBoundMass=None):
        def boundCalculation(simRes):
            intermediateSimRes = {"O_L": {},
                                  "O_H": {}}

            # TODO implement the ability of scoring other output names as "O"

            intermediateSimRes["O_L"]["1"] = simRes["O_L"]["1"]
            intermediateSimRes["O_L"]["0"] = simRes["O_H"]["0"]

            intermediateSimRes["O_H"]["1"] = simRes["O_H"]["1"]
            intermediateSimRes["O_H"]["0"] = simRes["O_L"]["0"]
            determineScores(intermediateSimRes)

            return intermediateSimRes

        #   [score(minOn_low, maxOff_high), score(minOn_high, maxOff_low)]

        intermediateSimRes = {key[:-2]: {"0": {},
                                         "1": {}}
                              for key in simRes.keys()}

        for key in intermediateSimRes:
            lowerIdent = key + "_L"
            upperIdent = key + "_H"
            intermediateSimRes[key]["0"] = {key: [] for key in simRes[lowerIdent]["0"].keys()}
            intermediateSimRes[key]["1"] = {key: [] for key in simRes[lowerIdent]["1"].keys()}

            levels = intermediateSimRes[key].keys()
            for level in levels:
                outputIdents = intermediateSimRes[key][level].keys()
                for outputIdent in outputIdents:
                    intermediateSimRes[key][level][outputIdent].append(simRes[lowerIdent][level][outputIdent])
                    intermediateSimRes[key][level][outputIdent].append(simRes[upperIdent][level][outputIdent])
                    intermediateSimRes[key][level][outputIdent] = np.array(intermediateSimRes[key][level][outputIdent])

        dist = simContext["dist"]

        for output in intermediateSimRes:
            currentScore = None
            currentOFF_CDF = None
            currentON_CDF = None

            if (dist == "ev-bound"):
                OFF_Data = intermediateSimRes[output]['0']
                ON_Data = intermediateSimRes[output]['1']
                OFF_Data = [OFF_Data[elem][1] for elem in OFF_Data]  # [1] Selects the Upper Envelope
                ON_Data = [ON_Data[elem][0] for elem in ON_Data]  # [0] Selects the Lower Envelope
                lowestOnL = np.min(ON_Data)
                highestOffH = np.max(OFF_Data)

                score = lowestOnL / highestOffH

                currentScore = score

                # if (currentScore == None or score < currentScore):
                #    currentScore = score
            else:
                currentLowerMassBounds = [entry[output] for entry in lowerBoundMass]

                intermediateSimRes[output]["0_CDF"] = {}
                OFF_CDFs = []
                for OFF_ident in intermediateSimRes[output]["0"]:
                    OFF_Bounds = intermediateSimRes[output]["0"][OFF_ident]
                    CDF = see.determineCDFNew(OFF_Bounds, currentLowerMassBounds)
                    OFF_CDFs.append(CDF)
                    intermediateSimRes[output]["0_CDF"][OFF_ident] = CDF

                intermediateSimRes[output]["1_CDF"] = {}
                ON_CDFs = []
                for ON_ident in intermediateSimRes[output]["1"]:
                    ON_Bounds = intermediateSimRes[output]["1"][ON_ident]
                    CDF = see.determineCDFNew(ON_Bounds, currentLowerMassBounds)
                    ON_CDFs.append(CDF)
                    intermediateSimRes[output]["1_CDF"][ON_ident] = CDF

                for OFF_CDF in OFF_CDFs:
                    for ON_CDF in ON_CDFs:
                        score = see.scoreEnvelopOutput(OFF_CDF=OFF_CDF, ON_CDF=ON_CDF, dist=dist)

                        if (currentScore == None or score < currentScore):
                            currentScore = score
                            currentOFF_CDF = OFF_CDF
                            currentON_CDF = ON_CDF

                if (simContext["visualise"]):
                    see.scoreEnvelopOutput(OFF_CDF=currentOFF_CDF, ON_CDF=currentON_CDF, dist=dist, visualise=True,
                                           PLOT_CDFs=sve.envelope_plotCDFs, PLOT_CDFs_DIFF=sve.envelope_plotCDFsDiff)

            intermediateSimRes[output]["SCORE"] = currentScore
        return intermediateSimRes

    envelopMode = simContext["envelop_mode"]
    # TODO: Assignment muss auch entsprechend auf Envelop_Mode umgestellt werden.

    nodeOrder = simData["circuit_valid_node_list"]
    circuit = simData["circuit"]
    originalCircuit = simData["circuit_original"]
    truthTable = simData["circuit_truthtable"]
    circuitVals = simData["circuit_simulation"]
    responseFunctions = simData["circuit_response_functions"]
    particles = simData["circuit_response_functions_particles"]

    envelopBoundingMass = simData["circuit_envelop_bounding_mass"]
    originalInputOutput = simData["circuit_original_input_output"]

    # insertNativeResponseFunctions(responseFunctions)
    insertNativeResponseFunctions(responseFunctions)

    if (envelopMode):
        nToUse = min([particles["max_number_of_particles"], simContext["numberOfParticles"]]) if (
            simContext["particles"]) else 1

        lowerBoundMass = None
        if (simContext["dist"] != "ev-bound"):
            lowerBoundMass = see.determineProbabilityMass(originalCircuit, envelopBoundingMass,
                                                          originalInputOutput=originalInputOutput,
                                                          originalAssignment=assignment, n=nToUse)
        assignment = see.parseAssignmentToEnvelopeAssignment(assignment)

    #    if (simContext["visualise_circuit"]):
    #        completeCircuitVals = {}

    truthTableToUse = truthTable
    if (simContext["use_custom_input_specification"]
            and (("custom_input_low" in simContext
                  and "custom_input_high" in simContext) or simContext["custom_input_specification"] != None)):

        inputSpecification = simContext["custom_input_specification"]
        if (inputSpecification == None):
            inputSpecification = generateInputSpecification(truthTable["input_IDs"], simContext["custom_input_low"],
                                                            simContext["custom_input_high"])
        updatedTruthTable = updateTruthTable(truthTable=truthTable, newInputSpecification=inputSpecification,
                                             envelopMode=envelopMode)
        truthTableToUse = updatedTruthTable

        simContext[
            "use_custom_input_specification"] = False  # Reset in order to enable explicit overwrite during simulation and automatic reconstruction of the initialisation setting if cis=0

    """
        Perform Simulation with the retrieved commands_Basic_Test.
    """

    simRes, completeCircuitVals = startSimulation(nodeOrder, circuit, assignment, truthTableToUse, responseFunctions,
                                                  particles,
                                                  dict(circuitVals))

    # debugPrint("simRes: \n" + str(simRes))
    """
        Determine Scores of all output variables
    """
    # print(simRes)

    if (envelopMode):
        simRes = determineEnvelopBasedScores(simRes, lowerBoundMass)
    else:
        determineScores(simRes)
    """
    for outID in simRes:
        for offID in simRes[outID]["0"]:
            print(offID, np.mean(simRes[outID]["0"][offID]), np.var(simRes[outID]["0"][offID]))
        for onID in simRes[outID]["1"]:
            print(onID, np.mean(simRes[outID]["1"][onID]), np.var(simRes[outID]["1"][onID]))
    """

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


def parseInput(inputText):
    def prepareString(prepText):
        prepText = " ".join(prepText.split())
        prepText = prepText.replace(" =", "=")
        prepText = prepText.replace("= ", "=")
        prepText = prepText.replace(" :", ":")
        prepText = prepText.replace(": ", ":")
        prepText = prepText.replace(" ,", ",")
        prepText = prepText.replace(", ", ",")
        return prepText

    def parseBool(inVal):
        curVal = inVal.lower();
        if (curVal == "0" or curVal == "false" or curVal == "f"):
            curVal = 0;
        elif (curVal == "1" or curVal == "true" or curVal == "t"):
            curVal = 1;
        return curVal

    inputText = prepareString(inputText)
    instruction = inputText.split()  # Split at Whitespace!
    # print(instruction)
    if (len(instruction) > 0):
        cmd = instruction[0]
        specDict = {}
        # print(inText)
        for elem in instruction[1:]:
            field, val = elem.split("=")
            # TODO Include all aspects which can be defined via interface ("s_path", "lib_path" etc. for example)
            # print(len(val), val, file=sys.stderr)
            # print(field, val)
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


def usermodePrint(text):
    if (simContext["usermode"]):
        print(text, file=sys.stdout)


def debugPrint(text):
    if (simContext["debug"]):
        print(text, file=sys.stderr)


def updateSimContext(specDict):
    for key in specDict:
        if (key in simContext):
            simContext[key] = specDict[key]


print("Start Simulator Initialisation")
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
simContext[
    "envelop_mode"] = False  # Can only be set at the initialisation of the simulator (if dist not set explicitly then ev-asym is used)
simContext["particle_quantiles"] = None
simContext["use_custom_input_specification"] = False  # Can only be set at the initialisation of the simulator
simContext["custom_input_specification"] = None
simContext["custom_input_low"] = 0.00001
simContext["custom_input_high"] = 10
simContext["wasserstein_p"] = 2.0
simContext["log_input_std"] = 0.  # have a clean input bei default
simContext["store_suffix"] = str(int(time.time()))

debugPrint(str(sys.argv))

cmd, specDict = parseInput(" ".join(sys.argv))
updateSimContext(specDict)

# Set the maximum number of particles as default
simContext["numberOfParticles"] = simContext["maxNumberOfParticles"];

if (simContext["structure"] != "NULL" and simContext["gate_lib"] != "NULL"):

    simData, simSpec = initialiseSimulator(simContext["structure"], simContext["gate_lib"],
                                           simContext["maxNumberOfParticles"])
    usermodePrint("Simulator Initialised and ready to use.")
    usermodePrint(
        "Dont forget to configure the assignment (append \"start\" with \"a_path=...\", the corresponding path, or directly by the assignment, by making use of\"assignment=...\"=  and the number of particles (append \"n=...\" to \"start\")")
    usermodePrint("Start simulation with \"start\"")
else:
    usermodePrint("You need to set the circuit structure and the library path to initialise the simulation!")
    usermodePrint(
        "This can be done by appending the command \"settings\" by \"s_path=...\" and the corresponding file path and next to this, the path to the gate lib with \"lib_path=...\"")
    structurePath = "NULL"
    gateLibPath = "NULL"
    configured = False
    inText = input("define settings:\n")
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

while (inText != "exit"):
    cmd, specDict = parseInput(inText)

    if (cmd == "start"):
        updateSimContext(specDict)

        if (simContext["assignment"] != "NULL"):
            # print(assignment)
            usermodePrint("Simulation started")
            startSimulation(assignment=simContext["assignment"], simData=simData, simSpec=simSpec)
            debugPrint("Iteration: " + str(iX))
            iX += 1
        else:
            print(
                "ERROR: Assignment needs to be defined at least once by adding \"assignment={...}\" or \"a_path=dir/file.json\" to after start.")

    if (cmd == "settings"):
        updateSimContext(specDict)

        simData, simSpec = initialiseSimulator(simContext["structure"], simContext["gate_lib"],
                                               simContext["maxNumberOfParticles"])
        usermodePrint("Simulator is reinitialised")

    inText = input("")
