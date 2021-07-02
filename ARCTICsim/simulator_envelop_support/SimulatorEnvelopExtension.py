import numpy as np

"""
@author: Erik Kubaczka
"""

VERSION = "SEE-3.1.4"


# The genetic gate lib consists of gates having an envelop representation included
# In order to be consistent with the equivalent envelop free circuit, the map of response functions should include a single transfer characteristic
# Therefore, the envelop representation is split up and each realisation is included explicitly with its upper and lower envelope in responseFunctions
# gateLib: The genetic gate library to prepare for usage within the equivalent envelope free circuit simulation
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
    # Iterate over each realisation in gateLib
    for gate in gateLib:
        # Gates not featuring an envelop representation can not be considered and are thus skipped
        # As a result, the implicit OR also requires an envelop representation.
        if (not "envelop_rep" in gate):
            continue;

        envelopeIdentifier = ["H", "L"]
        # Iterate over lower and upper envelope
        for ident in envelopeIdentifier:
            identifier = gate["identifier"] + "_" + ident

            functionIdentifier = ident + "_function"
            # Add the considered upper or lower envelope as an individual transfer characteristic to the response functions
            responseFunctions[identifier] = {"native": True,
                                             "parameters": dict.copy(
                                                 gate["envelop_rep"][functionIdentifier]["parameters"]),
                                             "library_parameters": dict.copy(
                                                 gate["envelop_rep"][functionIdentifier]["parameters"]),
                                             "envelop": True,
                                             # "type": "ENVELOPE"}
                                             "type": gate["envelop_rep"][functionIdentifier]["type"]}

            # It is possible to also include particles for an envelope. In case they are provided. they are copied here.
            if ("particles" in gate["envelop_rep"][functionIdentifier]):
                responseFunctions[identifier]["particle_parameters"] = list(
                    gate["envelop_rep"][functionIdentifier]["particles"].keys())

            # In order to enable the particle simulation, the response functions are inserted at simulation time
            # responseFunctions[identifier]["equation"] = getFunction(gate["envelop_rep"][functionIdentifier]["type"],
            #                                                         gate["envelop_rep"][functionIdentifier]["parameters"])

    return responseFunctions


# As in the "normal" case, the particles need to be prepared
def prepareEnvelopeParameters(particlesLib, gateLib, n):
    envelopeID = ["H", "L"]
    # Iterate over the single gates
    for gate in gateLib:
        for eID in envelopeID:
            # Create the identifier of the envelope and the corresponding transfer characteristic
            functionID = eID + "_function"
            gateIdentifier = gate["identifier"] + "_" + eID
            if ("particles" in gate["envelop_rep"][functionID]):
                particlesLib[gateIdentifier] = {}

                # Again get the particles for every parameter for which particles are provided
                for parameter in gate["envelop_rep"][functionID]['particles']:
                    rawParticles = gate["envelop_rep"][functionID]['particles'][parameter]

                    # Only check if the number of particles matches the required amount and do not create additional ones.
                    # This is the case, since the envelopes ideally should bound a probability mass,
                    # which in case the envelopes would be drawn randomly is unknown.
                    if (len(rawParticles) != n):
                        raise Exception("Insufficient number of parameters provided. (Provided: " + str(
                            len(rawParticles)) + ", Expected: " + str(n) + ")")

                    particles = np.array(rawParticles)

                    particlesLib[gateIdentifier][parameter] = particles

    return particlesLib


# Determines the probability mass the single realisations bound
def prepareEnvelopBoundingMass(gateLib):
    boundingMassValues = {}
    for gate in gateLib:
        gateIdentifier = gate["identifier"]
        # Only if the gate includes the required entry, it is added to the boundingMassValues Map
        if ("mass" in gate["envelop_rep"]):
            boundingMassValues[gateIdentifier] = {}
            boundingMassValues[gateIdentifier]["primitiveIdentifier"] = gate["primitiveIdentifier"]
            boundingMassValues[gateIdentifier]["mass"] = list.copy(gate["envelop_rep"]["mass"])

    return boundingMassValues


# This method translates the assignment in an assignment applicable to use within the equivalent envelope free circuit
# In principle new entries are generated which explicitly represent the envelopes.
# Consider the original entry is "NOR2_5" -> "A1_AmtR"
# The envelope assignment then includes the two entries "NOR2_5_H" -> "A1_AmtR_H" and "NOR2_5_L" -> "A1_AmtR_L"
def parseAssignmentToEnvelopeAssignment(assignment):
    envelopeAssignment = {}

    for key in assignment:
        envelopeAssignment[key + "_H"] = assignment[key] + "_H"
        envelopeAssignment[key + "_L"] = assignment[key] + "_L"

    return envelopeAssignment


# Transforms the envelope circuit (a normal circuit structure) into its equivalent envelope free pendant
def generateEnvelopCircuit(circuit, inputIDs, outputIDs, nodeDict):
    envelopCircuit = {}

    # The set of nodes where H is mapped to H and L to L, so no swap should be performed
    nodesToExclude = outputIDs
    # nodesToExclude.append("YFP")
    # Next to the output nodes, also for the OR no swap should be performed
    for elem in nodeDict:
        if (nodeDict[elem]["primitiveIdentifier"] == "OR2"):
            nodesToExclude.append(elem)
    nodesToExclude = set(nodesToExclude)

    # Determine the identifiers of the inputs in the equivalent envelop free circuit
    envelopInputIDs = []
    for key in inputIDs:
        envelopInputIDs.append(key + "_H")
        envelopInputIDs.append(key + "_L")

    # Determine the identifiers of the outputs in the equivalent envelop free circuit
    envelopOutputIDs = []
    for key in outputIDs:
        envelopOutputIDs.append(key + "_H")
        envelopOutputIDs.append(key + "_L")

    # Create the equivalent envelop free circuit based on the "normal" circuit representation
    for key in circuit:
        targets = circuit[key]
        # Each node is represented by two nodes in the new circuit
        envelopCircuit[key + "_H"] = []
        envelopCircuit[key + "_L"] = []
        for tar in targets:
            # Depending on the logic type of the target, a swap needs to be performed or not
            if (tar in nodesToExclude):
                envelopCircuit[key + "_H"].append(tar + "_H")
                envelopCircuit[key + "_L"].append(tar + "_L")
            else:
                envelopCircuit[key + "_H"].append(tar + "_L")
                envelopCircuit[key + "_L"].append(tar + "_H")

    # For debugging purposes the following enables to print the dot representation of the "normal" and the equivalent envelop free circuit
    plotDotRep = False
    if (plotDotRep):
        dotRepCircuit = circuitToDot(circuit, inputIDs, outputIDs)
        print(dotRepCircuit)

        dotRepEnvelopCircuit = circuitToDot(envelopCircuit, envelopInputIDs, envelopOutputIDs)
        print(dotRepEnvelopCircuit)

    # Return the equivalent envelope free circuit, the input ids and the output ids
    return envelopCircuit, envelopInputIDs, envelopOutputIDs

# Creates a CDF representation based on the provided bounds and the lower bound on the bounded probability mass
# bounds: The upper and lower bounds including the bounded probability mass
# lowerBounds: The lower bound on the probability mass included in the intervals
def determineCDFNew(bounds, lowerBounds):
    # Based on the log representation of the intervals a step distribution is created
    def determineStepDistribution(bounds_log, lowerBounds):

        n = len(bounds_log[0])  # Determine Number of particles Used
        U = np.zeros(n)
        diff = np.abs(bounds_log[1][0] - bounds_log[0][0])
        U[0] = lowerBounds[0] / diff
        m = diff * U[0]  # equal to "m = 1"

        # In case multiple intervals are provided, the probability mass included within them needs to be determined
        for i in range(1, n):
            diff = np.abs(bounds_log[1][i] - bounds_log[0][i])  # Calculate the interval width
            U[i] = lowerBounds[i] / diff
            if (U[i - 1] >= U[i]):                              # The height of an inner interval can not be smaller than the height of an enclosing intervall
                U[i] = U[i - 1]

            m += diff * (U[i] - U[i - 1])           # Sum up the normalizing constant

        U = U / m       # Normalize the distribution

        return U

    bounds_log = np.log(np.array(bounds))

    U = determineStepDistribution(bounds_log, lowerBounds)

    n = len(bounds_log[0])

    # Based on the step distribution the quantiles and their position is determined

    quantilesL = np.zeros(n)
    quantilesH = np.zeros(n)

    quantilesL[0] = 0
    quantilesH[0] = 1
    # First calculate the quantiles resulting from the lower and upper envelopes independently
    for i in range(1, n):
        quantilesL[i] = quantilesL[i - 1] + (bounds_log[0][i] - bounds_log[0][i - 1]) * U[i - 1]
        quantilesH[i] = quantilesH[i - 1] - (bounds_log[1][i - 1] - bounds_log[1][i]) * U[i - 1]

    # Combine the quantiles and position to a single CDF
    quantiles = np.concatenate((quantilesL, np.flip(quantilesH)))
    pos = np.concatenate((bounds_log[0], np.flip(bounds_log[1])))

    # Ensure that the calculated values fulfill the requirements
    for i in range(1, 2 * n):
        if (pos[i] < pos[i - 1]):
            raise Exception("Positions for CDF Creation are not ordered properly")

        if (quantiles[i] < quantiles[i - 1]):
            raise Exception("Resulting quantiles for CDF Creation are not ordered properly")

    return pos, quantiles


# OFF_CDF is a numpy array. OFF_CDF[0] contains the X positions and OFF_CDF[1] the Y values of the OFF CDF
# This is also the case for ON_CDF.
# It is required, that the values included in OFF_CDF[0] as well as in OFF_CDF[1], ON_CDF[0] and ON_CDF[1] are sorted ascending
def scoreEnvelopOutput(OFF_CDF, ON_CDF, dist="ev-asym", visualise=False, PLOT_CDFs=None, PLOT_CDFs_DIFF=None):
    # Based on the positions included in both CDFs an array including both positions is created
    def determinePositions(OFF_CDF_positions, ON_CDF_positions):
        # Combine both arrays and remove duplicates
        positions = np.unique(np.concatenate((OFF_CDF_positions, ON_CDF_positions)))
        # Sort in order to achieve an ascending positions list
        positions = np.sort(positions)
        return positions

    # Based on the original CDF defined on the range CDF_positions, a CDF defined on the values included in positions is calculated
    def determineCDF(positions, CDF_positions, CDF_values):
        pos = CDF_positions
        # pos = np.sort(np.array(pos))   # CDF_positions is sorted by construction

        CDF = np.zeros(len(positions))

        i = 0 # i is the index for the arrays CDF_position and CDF_values
        # Iterate over all positions included in positions
        for iP in range(len(positions)):
            # We need to obtain a value for the CDF at every position included in positions
            if (i < len(pos) and positions[iP] == pos[i]):
                # In case the asked position also exists within the provided CDF, simply use this position
                CDF[iP] = CDF_values[i]
                i += 1 # Increment i since the included position has already been considered

            else:  # Interpolate
                # In case the position is not already provided, we need to interpolate it based on the provided information
                if (i == 0):
                    # Since i is zero, the first value of CDF_values is used, which is also zero (required for a CDF starting point)
                    # i is zero if the asked position is before a position included in CDF_values
                    CDF[iP] = CDF_values[i]
                elif (i == len(pos)):
                    # The index i is not included in pos anymore so we are looking for a value behind the last value.
                    # This is equal to the last value in CDF_values and should be one (if the CDF is properly designed)
                    CDF[iP] = CDF_values[i - 1]
                else:
                    # The asked position is between two intermediate positions and thus we need to perform a true interpolation
                    x1 = pos[i - 1]
                    x2 = pos[i]
                    y1 = CDF_values[i - 1]
                    y2 = CDF_values[i]
                    CDF[iP] = interpolate((x1, y1), (x2, y2))[0](positions[iP])

        return CDF

    # CDF1 and CDF2 are both valid over positions, however there difference is not well defined
    # Therefore, this method adds values to positions, CDF1 and CDF2 such that whenever their difference has a zero crossing,
    # this zero crossing is explicitly included in the data.
    # This is especially relevant for a correct area respectively score calculation.
    def stuffCDFs(positions, CDF1, CDF2):
        # This method determines the intersection position of two straights defined by the four points
        def determineIntersectionPoint(x1, x2, y11, y12, y21, y22):
            denominator = y11 - y12 - y21 + y22
            x = (x2 * (y11 - y21) - x1 * (y12 - y22)) / denominator
            y = (y11 * y22 - y12 * y21) / denominator
            return (x, y)

        for i in range(len(positions) - 1):
            # Detect a zero crossing
            if (np.sign(CDF1[i] - CDF2[i]) * np.sign(CDF1[i + 1] - CDF2[i + 1]) == -1):
                # A transition happened, without an zero in the middle
                # Additional values need to be added
                p = determineIntersectionPoint(positions[i], positions[i + 1], CDF1[i], CDF1[i + 1], CDF2[i],
                                               CDF2[i + 1])
                # Insert the new position
                positions = np.concatenate((positions[:i + 1], np.ones(1) * p[0], positions[i + 1:]))
                CDF1 = np.concatenate((CDF1[:i + 1], np.ones(1) * p[1], CDF1[i + 1:]))
                CDF2 = np.concatenate((CDF2[:i + 1], np.ones(1) * p[1], CDF2[i + 1:]))

        return positions, CDF1, CDF2

    # Prepare the CDFs for the later score calculation.
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

    # Determine the sign of the score
    # Also the start and end of are calculation are set
    if (median1 > median2):
        start = median2
        end = median1
        sign = -1
    else:
        start = median1
        end = median2
        sign = 1

    # In case the symmetric scoring is used the start and end points are overwritten
    if (dist == "ev-sym"):
        start = min(positions)
        end = max(positions)

    # If a visualisation is requested, this visualisation is performed here
    if (visualise):
        PLOT_CDFs(positions, CDF1, CDF2, start, end)
        PLOT_CDFs_DIFF(positions, CDF1, CDF2, start, end)

    # The area between the two CDFs within range [start, end] is calculated
    area = determineArea(positions=positions, CDF1=CDF1, CDF2=CDF2, start=start, end=end)
    # The score is calculated based on the area and the sign
    score = np.exp(sign * area)
    return score


# This method creates a curve based quantiles (represented by pairs of percent) and determines the absolute area
# It is required that the first quantile is the 0% and the last is the 100% quantile
# positions: The positions the CDFs are defined over
# CDF1: The first CDF including values for all positions included in positions
# CDF2: The second CDF including values for all positions included in positions
# [start, end]: The interval over which the absolute area between the two CDFs is calculated
def determineArea(positions, CDF1, CDF2, start, end):
    # By already taking the absolute value of the difference into account later on, we directly obtain an absolute distance
    CDF_Diff = np.abs(np.array(CDF1) - np.array(CDF2))

    # Determine the indexes for positions such that the intervall [start, end] is included
    iStart = 0
    iEnd = len(positions) - 1

    while (start > positions[iStart]):
        iStart = iStart + 1

    while (end < positions[iEnd]):
        iEnd = iEnd - 1

    # Initialise the area with zero
    area = 0

    # In case start is smaller than the first considered entry in positions, the missing area needs to be added
    if (start < positions[iStart]):
        # Case distinction whether start is larger than the first entry in positions or not
        # The case distinction determines the height of the left and right side of the trapez
        if (iStart - 1 < 0):
            # If it is smaller than the first entry, which is the case for iStart == 0 and start < positions[iStart]
            h1 = CDF_Diff[0]
            h2 = CDF_Diff[0]
            pos1 = start
            pos2 = positions[iStart]
        else:
            # Smaller than an intermediate position
            # Interpolate the CDFs difference at this intermediate position and use this as height
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

        # Calculate and add the additional are
        additionalArea = (pos2 - pos1) * (h1 + h2) / 2
        area += additionalArea

    # Except the additional correction areas, the most part of the are can simply be calculated
    # by summing up the area of the trapezes included
    # TODO: Make use of CDF_Diff instead of explicitly calculating the differences again
    for i in range(iStart, iEnd):
        pos1 = positions[i]                     # The left position
        pos2 = positions[i + 1]                 # The right position
        h1 = np.abs(CDF1[i] - CDF2[i])          # The height at the left position
        h2 = np.abs(CDF1[i + 1] - CDF2[i + 1])  # The height at the right position

        # Calculate the area and add it
        additionalArea = 0.5 * (h1 + h2) * (pos2 - pos1)
        area += additionalArea

    # In case end is larger than the last considered entry in positions, the missing area needs to be added
    if (end > positions[iEnd]):
        # Case distinction whether end is larger than the last entry in positions or not
        # The case distinction determines the height of the left and right side of the trapez
        if (iEnd + 1 >= len(positions)):
            # If it is larger than the last entry, which is the case for iEnd + 1 >= len(positions) since end > positions[iEnd - 1]
            h1 = CDF_Diff[iEnd]
            h2 = CDF_Diff[iEnd]
            pos1 = positions[iEnd]
            pos2 = end
        else:
            # Larger than an intermediate position
            # Interpolate the CDFs difference at this intermediate position and use this as height
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

        # Calculate and add the additional are
        additionalArea = (pos2 - pos1) * (h1 + h2) / 2
        area += additionalArea


    return area

# A simple linear interpolation between two points
# Returns the linear function and its inverse representing this interpolation
def interpolate(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    m = (y2 - y1) / (x2 - x1)
    b = y2 - x2 * m

    return [(lambda x: m * x + b), (lambda y: (y - b) / m)]

# Calculates the median based on the CDF
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

            # Use the inverse of the linear interpolated function to get the median
            median = interpolate([x1, y1], [x2, y2])[1](0.5)

    return median



# This method calculates the pobability mass bounded by the envelopes at each gate
# For this, the probability mass is propagated through the circuit
#
# originalCircuit: The "normal" circuit corresponding to the equivalent envelope free circuit
# envelopBoundingMass: The probability mass each realisations envelopes bound
# originalInputOutput: The "normal" input and output ids
# originalAssignment: The "normal" assignment
# n: The number of particles per envelope (Respectively the number of envelope pairs per realisation)
def determineProbabilityMass(originalCircuit, envelopBoundingMass, originalInputOutput, originalAssignment, n):
    massValues = []

    # Comparable to a normal simulation the probability masses are propagated through the circuit
    # The "normal" circuit is required, since only scalar values are propagated

    # Iterate over the single envelope pairs
    for i in range(n):
        # Initialise the probability mass with 1
        circuitVals = {key: 1 for key in originalCircuit}

        # Insert the probability mass each realisation bounds with the respective envelope pair
        for nodeID in originalAssignment.keys():
            assignedRealisationID = originalAssignment[nodeID]
            circuitVals[nodeID] = envelopBoundingMass[assignedRealisationID]["mass"][i]


        nodeOrder = list(originalCircuit.keys())

        # Iterate over the circuit and propagate the probability mass from the inputs to the output
        for node in nodeOrder:
            destinationNodes = originalCircuit[node]
            for destNode in destinationNodes:
                circuitVals[destNode] = min(circuitVals[destNode], circuitVals[node])

        # Collect the value representing the probablity mass bounded by this circuit by using the respective envelope pairs
        massValues.append({outID: circuitVals[outID] for outID in originalInputOutput["output_IDs"]})

    return massValues

# Generates a dot representation of the circuit
# circuit: The circuit to visualize
# inputIDs: The input ids of the circuit to highlight as inputs
# outputIDs: The output ids of the circuit to highlight as outputs
def circuitToDot(circuit, inputIDs, outputIDs):
    dotRep = "strict digraph G {\n"
    # Group the inputs together
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

    # Group the outputs together
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

    # Add all gates to the graph
    for gateName in circuit:
        dotRep += "  " + gateName + "[label=" + gateName + "];\n"

    # Add the edges to the graph
    for gateName in circuit:
        for target in circuit[gateName]:
            dotRep += "  " + gateName + "->" + target + ";\n"

    dotRep += "}"

    # The dot representation as a string
    return dotRep


if __name__ == '__main__':
    print("Nothing is executed")
