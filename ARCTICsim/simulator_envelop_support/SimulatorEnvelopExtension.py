import numpy as np

"""
@author: Erik Kubaczka
"""

VERSION = "SEE-3.1.4"


def parseEnvelopeLibToGateLib(gateLib):
    def getFunction(type, parameters):
        def getInhibitoryHillFunction(parameters):
            ymax = parameters["ymax"]
            ymin = parameters["ymin"]
            K = parameters["K"]
            n = parameters["n"]

            function = (lambda x, p: ymin + (ymax - ymin) / (1 + (x / K) ** n))
            return function

        def getImplicitOrFunction(parameters):
            function = (lambda x, p: x)
            return function

        function = -1
        if ("INHIBITORY_HILL_EQUATION" == type):
            function = getInhibitoryHillFunction(parameters)
        elif ("IMPLICIT_OR" == type):
            function = getImplicitOrFunction(parameters)

        return function

    responseFunctions = {}
    for gate in gateLib:
        if (not "envelop_rep" in gate):
            continue;

        envelopeIdentifier = ["H", "L"]

        for ident in envelopeIdentifier:
            identifier = gate["identifier"] + "_" + ident

            functionIdentifier = ident + "_function"

            responseFunctions[identifier] = {"native": True,
                                             "parameters": dict.copy(
                                                 gate["envelop_rep"][functionIdentifier]["parameters"]),
                                             "library_parameters": dict.copy(
                                                 gate["envelop_rep"][functionIdentifier]["parameters"]),
                                             "envelop": True,
                                             # "type": "ENVELOPE"}
                                             "type": gate["envelop_rep"][functionIdentifier]["type"]}

            if ("particles" in gate["envelop_rep"][functionIdentifier]):
                responseFunctions[identifier]["particle_parameters"] = list(
                    gate["envelop_rep"][functionIdentifier]["particles"].keys())

            # In order to enable the particle simulation, the response functions are inserted at simulation time
            # responseFunctions[identifier]["equation"] = getFunction(gate["envelop_rep"][functionIdentifier]["type"],
            #                                                         gate["envelop_rep"][functionIdentifier]["parameters"])


    return responseFunctions


def prepareEnvelopeParameters(particlesLib, gateLib, n):
    envelopeID = ["H", "L"]
    for gate in gateLib:
        for eID in envelopeID:
            functionID = eID + "_function"
            gateIdentifier = gate["identifier"] + "_" + eID
            if ("particles" in gate["envelop_rep"][functionID]):
                particlesLib[gateIdentifier] = {}

                for parameter in gate["envelop_rep"][functionID]['particles']:
                    rawParticles = gate["envelop_rep"][functionID]['particles'][parameter]

                    if (len(rawParticles) != n):
                        raise Exception("Insufficient number of parameters provided. (Provided: " + str(
                            len(rawParticles)) + ", Expected: " + str(n) + ")")

                    particles = np.array(rawParticles)

                    particlesLib[gateIdentifier][parameter] = particles

    return particlesLib


def prepareEnvelopBoundingMass(gateLib):
    boundingMassValues = {}
    for gate in gateLib:
        gateIdentifier = gate["identifier"]
        if ("mass" in gate["envelop_rep"]):
            boundingMassValues[gateIdentifier] = {}
            boundingMassValues[gateIdentifier]["primitiveIdentifier"] = gate["primitiveIdentifier"]
            boundingMassValues[gateIdentifier]["mass"] = list.copy(gate["envelop_rep"]["mass"])

    return boundingMassValues


# start a_path=../Envelope_Test/structure_9_envelope_assignment.json
# start a_path=../Envelope_Test/structure_9_assignment.json
def parseAssignmentToEnvelopeAssignment(assignment):
    envelopeAssignment = {}

    for key in assignment:
        envelopeAssignment[key + "_H"] = assignment[key] + "_H"
        envelopeAssignment[key + "_L"] = assignment[key] + "_L"

    return envelopeAssignment


def generateEnvelopCircuit(circuit, inputIDs, outputIDs, nodeDict):
    envelopCircuit = {}

    nodesToExclude = outputIDs  # inputIDs + outputIDs
    #nodesToExclude.append("YFP")
    for elem in nodeDict:
        if (nodeDict[elem]["primitiveIdentifier"] == "OR2"):
            nodesToExclude.append(elem)
    nodesToExclude = set(nodesToExclude)



    envelopInputIDs = []
    for key in inputIDs:
        envelopInputIDs.append(key + "_H")
        envelopInputIDs.append(key + "_L")

    envelopOutputIDs = []
    for key in outputIDs:
        envelopOutputIDs.append(key + "_H")
        envelopOutputIDs.append(key + "_L")

    for key in circuit:
        targets = circuit[key]

        # if (key in nodesToExclude):
        #     envelopCircuit[key + "_H"] = [elem + "_H" for elem in targets]
        #     envelopCircuit[key + "_L"] = [elem + "_L" for elem in targets]
        # else:
        #
        # targets = set(targets)
        # excludedTargets = nodesToExclude.difference(targets)
        # normalTargets = targets.discard(nodesToExclude)
        # envelopCircuit[key + "_H"] = []
        # envelopCircuit[key + "_L"] = []
        # if (normalTargets != None):
        #     envelopCircuit[key + "_H"].append([elem + "_L" for elem in normalTargets])
        #     envelopCircuit[key + "_L"].append([elem + "_H" for elem in normalTargets])
        # if (excludedTargets != None):
        #     envelopCircuit[key + "_H"].append([elem + "_H" for elem in excludedTargets])
        #     envelopCircuit[key + "_L"].append([elem + "_L" for elem in excludedTargets])

        envelopCircuit[key + "_H"] = []
        envelopCircuit[key + "_L"] = []
        for tar in targets:
            if (tar in nodesToExclude):
                envelopCircuit[key + "_H"].append(tar + "_H")
                envelopCircuit[key + "_L"].append(tar + "_L")
            else:
                envelopCircuit[key + "_H"].append(tar + "_L")
                envelopCircuit[key + "_L"].append(tar + "_H")

    plotDotRep = False
    if (plotDotRep):
        dotRepCircuit = circuitToDot(circuit, inputIDs, outputIDs)
        print(dotRepCircuit)

        dotRepEnvelopCircuit = circuitToDot(envelopCircuit, envelopInputIDs, envelopOutputIDs)
        print(dotRepEnvelopCircuit)

    return envelopCircuit, envelopInputIDs, envelopOutputIDs




def determineCDFNew(bounds, lowerBounds):
    def determineStepDistribution(bounds_log, lowerBounds):

        n = len(bounds_log[0])  # Determine Number of particles Used
        U = np.zeros(n)
        diff = np.abs(bounds_log[1][0] - bounds_log[0][0])
        U[0] = lowerBounds[0] / diff
        m = diff * U[0]     # equal to "m = 1"
        for i in range(1, n):
            diff = np.abs(bounds_log[1][i] - bounds_log[0][i])
            U[i] = lowerBounds[i] / diff
            if (U[i - 1] >= U[i]):
                U[i] = U[i - 1]

            m += diff * (U[i] - U[i - 1])

        U = U / m

        return U

    bounds_log = np.log(np.array(bounds))

    U = determineStepDistribution(bounds_log, lowerBounds)

    n = len(bounds_log[0])
    
    quantilesL = np.zeros(n)
    quantilesH = np.zeros(n)
    
    quantilesL[0] = 0
    quantilesH[0] = 1

    for i in range(1, n):
        quantilesL[i] = quantilesL[i - 1] + (bounds_log[0][i] - bounds_log[0][i-1]) * U[i - 1]
        quantilesH[i] = quantilesH[i - 1] - (bounds_log[1][i - 1] - bounds_log[1][i]) * U[i - 1]


    quantiles = np.concatenate((quantilesL, np.flip(quantilesH)))
    pos = np.concatenate((bounds_log[0], np.flip(bounds_log[1])))

    for i in range(1, 2 * n):
        if (pos[i] < pos[i - 1]):
            raise Exception("Positions for CDF Creation are not ordered properly")

        if (quantiles[i] < quantiles[i - 1]):
            raise Exception("Resulting quantiles for CDF Creation are not ordered properly")

    return pos, quantiles


# OFF_CDF ist eine numpy array. OFF_CDF[0] enthält die X Punkte und OFF_CDF[1] die Y-Punkte der CDF von OFF
# Identisches gilt für ON_CDF.
# Es ist notwendig, dass die Werte in OFF_CDF[0] als auch OFF_CDF[1] aufsteigend sortiert sind.
def scoreEnvelopOutput(OFF_CDF, ON_CDF, dist="ev-asym", visualise=False, PLOT_CDFs=None, PLOT_CDFs_DIFF=None):
    def determinePositions(OFF_CDF_positions, ON_CDF_positions):
        positions = np.unique(np.concatenate((OFF_CDF_positions, ON_CDF_positions)))
        positions = np.sort(positions)
        return positions

    def determineCDF(positions, CDF_positions, CDF_values):
        pos = CDF_positions
        #pos = np.sort(np.array(pos))   # CDF_positions is sorted by construction

        CDF = np.zeros(len(positions))

        i = 0
        for iP in range(len(positions)):

            if (i < len(pos) and positions[iP] == pos[i]):
                CDF[iP] = CDF_values[i]
                i += 1

            else:  # Interpolate
                if (i == 0):
                    CDF[iP] = CDF_values[i]
                elif (i == len(pos)):
                    CDF[iP] = CDF_values[i - 1]
                else:
                    x1 = pos[i - 1]
                    x2 = pos[i]
                    y1 = CDF_values[i - 1]
                    y2 = CDF_values[i]
                    CDF[iP] = interpolate((x1, y1), (x2, y2))[0](positions[iP])

        return CDF

    def stuffCDFs(positions, CDF1, CDF2):
        def determineIntersectionPoint(x1, x2, y11, y12, y21, y22):
            denominator = y11 - y12 - y21 + y22
            x = (x2 * (y11 - y21) - x1 * (y12 - y22)) / denominator
            y = (y11 * y22 - y12 * y21) / denominator
            return (x, y)


        for i in range(len(positions) - 1):
            if (np.sign(CDF1[i] - CDF2[i]) * np.sign(CDF1[i + 1] - CDF2[i + 1]) == -1):
                # A transition happened, without an zero in the middle
                # Additional values need to be added
                p = determineIntersectionPoint(positions[i], positions[i + 1], CDF1[i], CDF1[i + 1], CDF2[i],
                                               CDF2[i + 1])

                positions = np.concatenate((positions[:i + 1], np.ones(1) * p[0], positions[i + 1:]))
                CDF1 = np.concatenate((CDF1[:i + 1], np.ones(1) * p[1], CDF1[i + 1:]))
                CDF2 = np.concatenate((CDF2[:i + 1], np.ones(1) * p[1], CDF2[i + 1:]))

        return positions, CDF1, CDF2


    OFF_CDF_pos = OFF_CDF[0]
    OFF_CDF_CDF = OFF_CDF[1]
    ON_CDF_pos = ON_CDF[0]
    ON_CDF_CDF = ON_CDF[1]
    positions = determinePositions(OFF_CDF_pos, ON_CDF_pos)
    CDF1 = determineCDF(positions, CDF_positions=OFF_CDF_pos, CDF_values=OFF_CDF_CDF)
    CDF2 = determineCDF(positions, CDF_positions=ON_CDF_pos, CDF_values=ON_CDF_CDF)
    positions, CDF1, CDF2 = stuffCDFs(positions, CDF1, CDF2)

    median1 = determineMedian(positions, CDF1)
    median2 = determineMedian(positions, CDF2)

    if (median1 > median2):
        start = median2
        end = median1
        sign = -1
    else:
        start = median1
        end = median2
        sign = 1

    if (dist == "ev-sym"):
        start = min(positions)
        end = max(positions)

    if (visualise):
        PLOT_CDFs(positions, CDF1, CDF2, start, end)
        PLOT_CDFs_DIFF(positions, CDF1, CDF2, start, end)

    area = determineArea(positions=positions, CDF1=CDF1, CDF2=CDF2, start=start, end=end)
    score = np.exp(sign * area)
    return score


# Setzt aus den Quantilen (welche Wert prozent paare darstellen) eine Kurve zusammen und berechnet deren Wert
# Erstes element muss 0% Quantil sein und das letzte das 100% Quantil.
# Im Idealfall ist das 50% Quantil gegeben
def determineArea(positions, CDF1, CDF2, start, end):
    CDF_Diff = np.abs(np.array(CDF1) - np.array(CDF2))

    iStart = 0
    iEnd = len(positions) - 1
    while (start > positions[iStart]):
        iStart = iStart + 1

    while (end < positions[iEnd]):
        iEnd = iEnd - 1

    area = 0

    if (start < positions[iStart]):
        if (iStart - 1 < 0):
            h1 = CDF_Diff[0]
            h2 = CDF_Diff[0]
            pos1 = start
            pos2 = positions[iStart]
        else:
            y1 = CDF_Diff[iStart - 1]
            y2 = CDF_Diff[iStart]
            x1 = positions[iStart - 1]
            x2 = positions[iStart]

            # m = (y2 - y1) / (x2 - x1)
            # b = y2 - x2 * m

            pos1 = start
            pos2 = x2

            # h1 = m * pos1 + b
            h1 = interpolate([x1, y1], [x2, y2])[0](pos1)
            h2 = y2

        additionalArea = (pos2 - pos1) * (h1 + h2) / 2
        area += additionalArea

    for i in range(iStart, iEnd):
        pos1 = positions[i]
        pos2 = positions[i + 1]
        h1 = np.abs(CDF1[i] - CDF2[i])
        h2 = np.abs(CDF1[i + 1] - CDF2[i + 1])

        additionalArea = 0.5 * (h1 + h2) * (pos2 - pos1)
        area += additionalArea

    if (end > positions[iEnd]):
        if (iEnd + 1 >= len(positions)):
            h1 = CDF_Diff[iEnd]
            h2 = CDF_Diff[iEnd]
            pos1 = positions[iEnd]
            pos2 = end
        else:
            y1 = CDF_Diff[iEnd]
            y2 = CDF_Diff[iEnd + 1]
            x1 = positions[iEnd]
            x2 = positions[iEnd + 1]

            # m = (y2 - y1) / (x2 - x1)
            # b = y2 - x2 * m

            pos1 = x1
            pos2 = end

            h1 = y1
            # h2 = m * pos2 + b
            h2 = interpolate([x1, y1], [x2, y2])[0](pos2)

        additionalArea = (pos2 - pos1) * (h1 + h2) / 2
        area += additionalArea

    # sign = determineSign(positions, CDF1, CDF2)
    # area = sign * area
    return area


def interpolate(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    m = (y2 - y1) / (x2 - x1)
    b = y2 - x2 * m

    return [(lambda x: m * x + b), (lambda y: (y - b) / m)]


def determineMedian(positions, CDF):
    median = None
    for i in range(len(CDF) - 1):
        if (CDF[i] == 0.5):
            median = positions[i]

        elif (CDF[i] < 0.5 and CDF[i + 1] > 0.5):
            x1 = positions[i]
            y1 = CDF[i]
            x2 = positions[i + 1]
            y2 = CDF[i + 1]

            median = interpolate([x1, y1], [x2, y2])[1](0.5)

    return median


# def determineSign(positions, CDF1, CDF2):
#    median1 = determineMedian(positions, CDF1)
#    median2 = determineMedian(positions, CDF2)

#    sign = 1 if (median2 > median1) else -1
#    return sign

# Methode zur Berechnung der m_e entsprechend der Vorgabe
# (Vorgabe Nicolai) Algorithmus: Lege an jedem Input den Wert 1 an. Rechne den (normalen) Schaltkreis, wobei folgende Substitutionen gemacht werden. Jedes NOR-Gate mit inputs x_0, x_1 und output y wird gegen die Operation y = x_0*x_1*g_e getauscht. Jedes NOT-Gate mit input x und output y wird mit der Operation y = x*g_e getauscht. g_e wird hier aus der Library gelesen, und zwar zu jedem Envelope-Partikel individuell. Das heisst, es gibt Eintraege 'envelop_rep' -> 'mass", wobei 'mass' eine List derselben Laenge wie die Partikel ist und das jeweilige g_e fuer diesen Partikeldurchlauf enthaelt
def determineProbabilityMassOldVersion(originalCircuit, envelopBoundingMass, originalInputOutput, originalAssignment, n):
    massValues = []

    for i in range(n):
        circuitVals = {key: 1 for key in originalCircuit}
        # for inputID in originalInputOutput["input_IDs"]:
        #    circuitVals[inputID] = 1

        for nodeID in originalAssignment.keys():
            assignedRealisationID = originalAssignment[nodeID]
            circuitVals[nodeID] = envelopBoundingMass[assignedRealisationID]["mass"][i]

        # for outputID in originalInputOutput["output_IDs"]:
        #    circuitVals[outputID] = 1

        nodeOrder = list(originalCircuit.keys())

        for node in nodeOrder:
            destinationNodes = originalCircuit[node]
            for destNode in destinationNodes:
                circuitVals[destNode] *= circuitVals[node]

        massValues.append({outID: circuitVals[outID] for outID in originalInputOutput["output_IDs"]})

    return massValues


def determineProbabilityMass(originalCircuit, envelopBoundingMass, originalInputOutput, originalAssignment, n):
    massValues = []

    for i in range(n):
        circuitVals = {key: 1 for key in originalCircuit}
        # for inputID in originalInputOutput["input_IDs"]:
        #    circuitVals[inputID] = 1

        for nodeID in originalAssignment.keys():
            assignedRealisationID = originalAssignment[nodeID]
            circuitVals[nodeID] = envelopBoundingMass[assignedRealisationID]["mass"][i]

        # for outputID in originalInputOutput["output_IDs"]:
        #    circuitVals[outputID] = 1

        nodeOrder = list(originalCircuit.keys())

        for node in nodeOrder:
            destinationNodes = originalCircuit[node]
            for destNode in destinationNodes:
                circuitVals[destNode] = min(circuitVals[destNode], circuitVals[node])

        massValues.append({outID: circuitVals[outID] for outID in originalInputOutput["output_IDs"]})

    return massValues


def circuitToDot(circuit, inputIDs, outputIDs):
    dotRep = "strict digraph G {\n"
    dotRep += "subgraph inputs {\n"
    for inID in inputIDs:
        dotRep += "  " + inID + "[label=" + inID + "];\n"
        """
        dotRep += "subgraph " + inID + " { \n"
        dotRep += "  " + inID + "_H [label=" + inID + "_H];\n"
        dotRep += "  " + inID + "_L [label=" + inID + "_L];\n"
        dotRep += "}\n"
        """
    dotRep += "}\n"

    dotRep += "subgraph outputs {\n"
    for outID in outputIDs:
        dotRep += "  " + outID + "[label=" + outID + "];\n"
        """
        dotRep += "subgraph " + inID + " { \n"
        dotRep += "  " + inID + "_H [label=" + inID + "_H];\n"
        dotRep += "  " + inID + "_L [label=" + inID + "_L];\n"
        dotRep += "}\n"
        """
    dotRep += "}\n"

    for gateName in circuit:
        dotRep += "  " + gateName + "[label=" + gateName + "];\n"

    for gateName in circuit:
        for target in circuit[gateName]:
            dotRep += "  " + gateName + "->" + target + ";\n"

    dotRep += "}"

    return dotRep


if __name__ == '__main__':
    print("Nothing is executed")
