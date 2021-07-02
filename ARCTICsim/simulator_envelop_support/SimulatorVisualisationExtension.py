import matplotlib.pyplot as plt
import numpy as np

# import os

"""
@author: Erik Kubaczka
"""

"""
Attention: Response Functions are only correct for the last provided parameter -> Particles not supported
"""

# Calls plotCircuitWithValues for each possible input assignment and thus yields a visualisation of every combination possible
def visualiseCircuitWithValues(circuit, assignment, responseFunctions, circuitValues):
    for inputID in circuitValues:
        plotCircuitWithValues(inputID, circuit=circuit, assignment=assignment, responseFunctions=responseFunctions,
                              circuitVals=circuitValues[inputID])

    return


# Visualizes the transfer characteristic of the gates in combination with the concentrations present within the circuit
# The plot is saved to .svg and .png
def plotCircuitWithValues(inputID, circuit, assignment, responseFunctions, circuitVals):
    # Determines the single input values of a gate.
    # In case of a NOR, two input values exist, while they can not be reconstructed from the gates output values
    def getGateInputValues(gate, circuit, circuitVals):
        inputValues = []
        for src in circuit:
            if (gate in circuit[src]):
                inputValues.append(circuitVals[src])

        gateValue = circuitVals[gate]

        # The input values as a list, their sum as well as the corresponding output of the gate
        return inputValues, sum(inputValues), gateValue

    # Define the definition domain of interest
    X = np.logspace(-4, 2, 100)
    # Determine the dimensions of the subplot to adequatly arange all plots
    plotDimensions = estimateDimensionsOfSubplots(len(assignment))

    plt.figure()
    fig, ax = plt.subplots(int(plotDimensions[0]), int(plotDimensions[1]))
    fig.subplots_adjust(wspace=1, hspace=1)
    i = 0
    # Iterate over the single gates (input buffers, logic gates and output buffers) of a circuit
    for elem in circuit:
        currentAxis = ax[int(i / plotDimensions[1])][int(i % plotDimensions[1])]
        # plt.subplot(plotDimensions)
        if (elem not in assignment):  # non logic gates are skipped since there is no transfer characteristic to show
            continue

        # Plot the corresponding responsefunction
        responseFunction = responseFunctions[assignment[elem]]
        Y = plotResponseFunction(ax=currentAxis, X=X, responseFunction=responseFunction["equation"],
                                 parameters=responseFunction["parameters"])

        # Get the gates input values
        inputValues, inputVal, outputVal = getGateInputValues(elem, circuit, circuitVals)
        # Check if multiple input values exist and plot them if so
        if (len(inputValues) > 1):
            for inVal in inputValues:
                currentAxis.plot([inVal, inVal], [min(Y), outputVal], "gx")

        # Plot the mapping from cummulative input via transition point to output value
        currentAxis.plot([inputVal, inputVal, min(X)], [min(Y), outputVal, outputVal], "r.")

        currentAxis.set_title(elem)  # + " (" + assignment[elem] + ")")
        currentAxis.set_xscale("log")
        currentAxis.set_yscale("log")
        # currentAxis.set_ylim(bottom=min(Y) / 4)
        currentAxis.set_ylim(bottom=min(X), top=max(X))

        i += 1
    # print(os.path.abspath("visualisation/" + inputID + ".svg"))
    # Save the plot
    fig.suptitle("Circuit for Input:" + inputID)
    plt.savefig("visualisation/" + inputID + ".svg")
    plt.savefig("visualisation/" + inputID + ".png")
    plt.close(fig)
    # plt.show()

# Adds the response function (specified by responseFunction and parameters) sampled at positions X to the provided axes (ax)
def plotResponseFunction(ax, X, responseFunction, parameters):
    Y = np.zeros(len(X))
    for i in range(len(X)):
        Y[i] = responseFunction(X[i], parameters)

    ax.plot(X, Y)

    return Y

# Determines the number of subplots required for adequatly visualizing the circuit.
def estimateDimensionsOfSubplots(n):
    squareRoot = np.sqrt(n)
    dim = np.zeros(2)
    dim[0] = np.ceil(squareRoot)
    dim[1] = np.floor(n / dim[0])

    i = 1
    while (dim[0] * dim[1] < n):
        dim[i] = dim[i] + 1
        i = 1 - i
    return dim

# Plots two CDFs in the same graph
# Thereby CDF1 and CDF2 need to be defined for the values given in positions
# positions: The X-axes value
# CDF1: The first CDF
# CDF2: The second CDF
# [start, end]: The interval of the area between the two CDFs, which is taken into account for scoring purposes
def envelope_plotCDFs(positions, CDF1, CDF2, start, end):
    plt.figure()
    plt.plot(positions, CDF1, label="CDF1")
    plt.plot(positions, CDF2, label="CDF2")
    plt.plot(np.ones(2) * start, [0, 1], "--", label="Start")
    plt.plot(np.ones(2) * end, [0, 1], "--", label="End")
    plt.legend()
    plt.title("The CDFs")
    plt.show()


# Plots the difference between two CDFs
# Thereby CDF1 and CDF2 need to be defined for the values given in positions
# positions: The X-axes value
# CDF1: The first CDF
# CDF2: The second CDF
# [start, end]: The interval of the area between the two CDFs, which is taken into account for scoring purposes
def envelope_plotCDFsDiff(positions, CDF1, CDF2, start, end):
    cdf1 = np.array(CDF1)
    cdf2 = np.array(CDF2)

    cdf_diff = cdf1 - cdf2

    plt.figure()
    plt.plot(positions, cdf_diff, label="CDF Difference")
    plt.plot(positions, np.abs(cdf_diff), label="Absolute Difference")
    plt.plot(np.ones(2) * start, [0, 1], "--", label="Start")
    plt.plot(np.ones(2) * end, [0, 1], "--", label="End")
    plt.legend()
    plt.title("Difference of CDFs")
    plt.show()
