import matplotlib.pyplot as plt
import numpy as np

"""
Attention: Response Functions are only correct for the last provided parameter -> Particles not supported
"""


def visualiseCircuitWithValues(circuit, assignment, responseFunctions, circuitValues):
    for inputID in circuitValues:
        plotCircuitWithValues(inputID, circuit=circuit, assignment=assignment, responseFunctions=responseFunctions,
                              circuitVals=circuitValues[inputID])

    return


def plotCircuitWithValues(inputID, circuit, assignment, responseFunctions, circuitVals):
    def getGateInputValues(gate, circuit, circuitVals):
        inputValues = []
        for src in circuit:
            if (gate in circuit[src]):
                inputValues.append(circuitVals[src])

        gateValue = circuitVals[gate]

        return inputValues, sum(inputValues), gateValue

    X = np.logspace(-5, 2, 100)
    plotDimensions = estimateDimensionsOfSubplots(len(assignment))

    plt.figure()
    fig, ax = plt.subplots(int(plotDimensions[0]), int(plotDimensions[1]))
    fig.subplots_adjust(wspace=1, hspace=1)
    i = 0
    for elem in circuit:
        currentAxis = ax[int(i / plotDimensions[1])][int(i % plotDimensions[1])]
        # plt.subplot(plotDimensions)
        if (elem not in assignment):
            continue

        responseFunction = responseFunctions[assignment[elem]]
        Y = plotResponseFunction(ax=currentAxis, X=X, responseFunction=responseFunction["equation"],
                                 parameters=responseFunction["parameters"])

        inputValues, inputVal, outputVal = getGateInputValues(elem, circuit, circuitVals)
        if (len(inputValues) > 1):
            for inVal in inputValues:
                currentAxis.plot([inVal, inVal], [min(Y), outputVal], "gx")

        currentAxis.plot([inputVal, inputVal, min(X)], [min(Y), outputVal, outputVal], "r.")

        currentAxis.set_title(elem)  # + " (" + assignment[elem] + ")")
        currentAxis.set_xscale("log")
        currentAxis.set_yscale("log")
        currentAxis.set_ylim(bottom=min(Y) / 4)

        i += 1

    fig.suptitle("Circuit for Input:" + inputID)
    plt.savefig("visualisation/" + inputID + ".svg")
    plt.savefig("visualisation/" + inputID + ".png")
    plt.close(fig)
    #plt.show()


def plotResponseFunction(ax, X, responseFunction, parameters):
    Y = np.zeros(len(X))
    for i in range(len(X)):
        Y[i] = responseFunction(X[i], parameters)

    ax.plot(X, Y)

    return Y


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



def envelope_plotCDFs(positions, CDF1, CDF2, start, end):
    plt.figure()
    plt.plot(positions, CDF1, label="CDF1")
    plt.plot(positions, CDF2, label="CDF2")
    plt.plot(np.ones(2) * start, [0, 1], "--", label="Start")
    plt.plot(np.ones(2) * end, [0, 1], "--", label="End")
    plt.legend()
    plt.title("The CDFs")
    plt.show()

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

