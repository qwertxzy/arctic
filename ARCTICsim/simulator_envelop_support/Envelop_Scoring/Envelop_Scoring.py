import scipy.stats as st
import numpy as np
import matplotlib.pyplot as plt


def plotHists(OFF_Data, ON_Data):
    plt.figure()

    # bins = np.logspace(-3, 2, 100);
    bins = np.linspace(10 ** -3, 10 ** 1, 100);

    plt.hist(OFF_Data, bins=bins, label="OFF")
    plt.hist(ON_Data, bins=bins, label="ON")
    # plt.gca().set_xscale("log")
    plt.legend()
    plt.show()


def plotCDF(OFF_Data, ON_Data):
    OFF_Data_sorted = np.sort(OFF_Data)
    ON_Data_sorted = np.sort(ON_Data)

    OFF_CDF = np.array(range(1, len(OFF_Data_sorted) + 1)) / len(OFF_Data_sorted)
    ON_CDF = np.array(range(1, len(ON_Data_sorted) + 1)) / len(ON_Data_sorted)

    OFF_median = np.median(OFF_Data)
    ON_median = np.median(ON_Data)

    plt.figure()
    plt.plot(OFF_Data_sorted, OFF_CDF, color="orange", label="OFF")
    plt.plot(np.ones(2) * OFF_median, [0, 1], "--", color="orange", label="OFF Median")

    plt.plot(ON_Data_sorted, ON_CDF, "b", label="ON")
    plt.plot(np.ones(2) * ON_median, [0, 1], "--", color="blue", label="ON Median")

    plt.legend()
    plt.show()


def plotHistAndHistOfLogData(data):
    plt.figure()

    plt.hist(data)
    plt.hist(np.log(data))

    plt.show()


def scoringFunctionSymmetric(OFF_Bounds, ON_Bounds):
    # Werte logarithmieren und abschließend exponentieren

    distA = np.abs(ON_Bounds[0] - OFF_Bounds[0])
    distB = np.abs(ON_Bounds[1] - OFF_Bounds[1])

    if (ON_Bounds[1] >= OFF_Bounds[1] and ON_Bounds[0] >= OFF_Bounds[0]):
        # Die ON CDF ist immer rechts von der OFF CDF
        score = (distA + distB) / 2

    elif (ON_Bounds[1] < OFF_Bounds[1] and ON_Bounds[0] < OFF_Bounds[0]):
        # Die ON CDF ist immer links der OFF CDF
        score = - (distA + distB) / 2

    else:
        # Die CDFs überschneiden sich

        sign = 1 if (ON_Bounds[0] + ON_Bounds[1] > OFF_Bounds[0] + OFF_Bounds[1]) else -1

        score = 0.5 * (distA ** 2 + distB ** 2) / (distA + distB)
        score = sign * score

    return score


def scoringFunctionAsymmetric(OFF_Bounds, ON_Bounds):
    def scoreIntersectingCDFs(interval1, interval2, sign):
        a1, b1 = interval1
        a2, b2 = interval2

        median1 = (a1 + b1) / 2
        median2 = (a2 + b2) / 2

        width1 = median1 - a1
        width2 = b2 - median2

        CDF_1 = lambda x: (x - a1) / (b1 - a1) if (
                x >= a1 and x <= b1) else (1 if (x > b1) else 0)

        CDF_2 = lambda x: (x - a2) / (b2 - a2) if (
                x >= a2 and x <= b2) else (1 if (x > b2) else 0)

        h1 = 0.5 - CDF_2(median1)
        h2 = CDF_1(median2) - 0.5

        w1 = max(0, a2 - median1)
        w3 = max(0, median2 - b1)
        w2 = median2 - median1 - w1 - w3

        area1 = w1 * h1
        area2 = w3 * h2
        area3 = (w1 + w2) * h2 / 2
        area4 = (w2 + w3) * h1 / 2

        score = area1 + area2 + area3 + area4
        score = sign * score

        return score

    # Werte logarithmieren und abschließend exponentieren

    if (np.sum(OFF_Bounds) <= np.sum(ON_Bounds)):
        # Die ON CDF ist immer rechts von der OFF CDF
        # Zur Anpassung an die Diracs muss lediglich die Fläche der Dreiecke entfernt werden

        a1 = OFF_Bounds[0]
        b1 = OFF_Bounds[1]

        a2 = ON_Bounds[0]
        b2 = ON_Bounds[1]

        sign = 1
    else:
        a1 = ON_Bounds[0]
        b1 = ON_Bounds[1]

        a2 = OFF_Bounds[0]
        b2 = OFF_Bounds[1]

        sign = -1

    # if (ON_Bounds[1] >= OFF_Bounds[1] and ON_Bounds[0] >= OFF_Bounds[0]
    # or ON_Bounds[1] <= OFF_Bounds[1] and ON_Bounds[0] <= OFF_Bounds[0]):
    # score = scoreNonIntersectingCDFs([a1, b1], [a2, b2], sign)
    # else:
    score = scoreIntersectingCDFs([a1, b1], [a2, b2], sign)

    return score


def normalWasserstein(OFF_Data, ON_Data):
    sign = 1 if (np.median(ON_Data) > np.median(OFF_Data)) else -1
    score = st.wasserstein_distance(OFF_Data, ON_Data)
    score = sign * score

    return score


def asymmetricWasserstein(OFF_Data, ON_Data):
    sign = 1 if (np.median(ON_Data) > np.median(OFF_Data)) else -1
    median_on = np.median(ON_Data)
    median_off = np.median(OFF_Data)
    # if (median_on > median_off):
    #    ON_Data[ON_Data > median_on] = median_on
    #    OFF_Data[OFF_Data < median_off] = median_off
    # else:
    #    ON_Data[ON_Data <= median_on] = median_on
    #    OFF_Data[OFF_Data >= median_off] = median_off
    score = st.wasserstein_distance(ON_Data, OFF_Data)
    score = sign * score

    return score


def testUniformScoring():
    N = 10 ** 5

    # OFF_Bounds = [0, 4]
    # ON_Bounds = [1.5, 2.5]

    # OFF_Bounds = [0, 4]
    # ON_Bounds = [3, 4]

    # OFF_Bounds = [0, 2]
    # ON_Bounds = [3, 4]

    # OFF_Bounds = [0, 5]
    # ON_Bounds = [1.5, 2.8]

    OFF_Bounds = [0, 5]
    ON_Bounds = [4, 7]

    # OFF_Bounds = [0, 6]
    # ON_Bounds = [1, 3]

    OFF_Data = np.random.uniform(OFF_Bounds[0], OFF_Bounds[1], N)
    ON_Data = np.random.uniform(ON_Bounds[0], ON_Bounds[1], N)

    OFF_Data1 = np.random.uniform(OFF_Bounds[0], OFF_Bounds[1], N)
    OFF_Data2 = np.random.uniform(OFF_Bounds[0] * 2, OFF_Bounds[1] / 2, N)
    OFF_Data = np.concatenate((OFF_Data1, OFF_Data2))
    ON_Data = np.concatenate((np.random.uniform(ON_Bounds[0], ON_Bounds[1], N),
                              np.random.uniform(ON_Bounds[0] * 2, ON_Bounds[1] / 2, N)))

    scoreSymmetric = normalWasserstein(OFF_Data, ON_Data)
    scoreAsymmetric = asymmetricWasserstein(OFF_Data, ON_Data)

    calculatedScoreSymmetric = scoringFunctionSymmetric(OFF_Bounds, ON_Bounds)
    calculatedScoreAsymmetric = scoringFunctionAsymmetric(OFF_Bounds, ON_Bounds)

    # plotHists(OFF_Data, ON_Data)
    plotCDF(OFF_Data, ON_Data)
    print("Symmetric:")
    print("Estimated:", scoreSymmetric)
    print("Calculated:", calculatedScoreSymmetric)

    print("Asymmetric:")
    print("Estimated:", scoreAsymmetric)
    print("Calculated:", calculatedScoreAsymmetric)


def plotQuantiles(positions, CDF1, CDF2, start, end):
    plt.figure()
    plt.plot(positions, CDF1, label="CDF1")
    plt.plot(positions, CDF2, label="CDF2")
    plt.plot(np.ones(2) * start, [0, 1], "--", label="Start")
    plt.plot(np.ones(2) * end, [0, 1], "--", label="End")
    plt.legend()
    plt.title("The CDFs")
    plt.show()


def plotQuantileDifference(positions, CDF1, CDF2, start, end):
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


def determinePositions(OFF_Bounds, ON_Bounds):
    positions = list(set(list(OFF_Bounds[0]) + list(OFF_Bounds[1]) + list(ON_Bounds[0]) + list(ON_Bounds[1])))
    positions = np.array(positions)
    positions = np.sort(positions)
    return positions


def determineCDF(positions, bounds, quantiles):
    pos = list(bounds[0]) + list(bounds[1])
    pos = np.sort(np.array(pos))

    CDF = np.zeros(len(positions))

    i = 0
    for iP in range(len(positions)):

        if (i < len(pos) and positions[iP] == pos[i]):
            CDF[iP] = quantiles[i]
            i += 1

        else:  # Interpolate
            if (i == 0):
                CDF[iP] = quantiles[i]
            elif (i == len(pos)):
                CDF[iP] = quantiles[i - 1]
            else:
                x1 = pos[i - 1]
                x2 = pos[i]
                y1 = quantiles[i - 1]
                y2 = quantiles[i]
                CDF[iP] = interpolate((x1, y1), (x2, y2))[0](positions[iP])

    return CDF


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


# Setzt aus den Quantilen (welche Wert prozent paare darstellen) eine Kurve zusammen und berechnet deren Wert
# Erstes element muss 0% Quantil sein und das letzte das 100% Quantil.
# Im Idealfall ist das 50% Quantil gegeben
def determineArea(positions, cdf1, cdf2, start, end):
    def determineSign(positions, CDF1, CDF2):
        median1 = determineMedian(positions, CDF1)
        median2 = determineMedian(positions, CDF2)

        sign = 1 if (median2 > median1) else -1
        return sign

    def stuffCDFs(positions, CDF1, CDF2):
        def determineIntersectionPoint(x1, x2, y11, y12, y21, y22):
            denominator = y11 - y12 - y21 + y22
            x = (x2 * (y11 - y21) - x1 * (y12 - y22)) / denominator
            y = (y11 * y22 - y12 * y21) / denominator
            return (x, y)

        # diff = np.array(CDF1) - np.array(CDF2)
        # diffSign = np.sign(diff)
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

    CDF1 = np.array(cdf1)
    CDF2 = np.array(cdf2)
    positions, CDF1, CDF2 = stuffCDFs(positions, CDF1, CDF2)
    plotQuantileDifference(positions, CDF1=CDF1, CDF2=CDF2, start=start, end=end)

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

    sign = determineSign(positions, CDF1, CDF2)
    area = sign * area
    return area


def testQuantileBasedScoring():
    positions = [1, 2, 3, 4, 5, 6, 7]
    CDF1 = [0, 0.25, 0.5, 0.75, 1, 1, 1]
    # CDF2 = [0, 0.25, 0.75, 1, 1, 1]
    CDF2 = [0, 0, 1, 1, 1, 1, 1]
    start = 0
    end = 6
    plotQuantiles(positions, CDF1, CDF2, start, end)
    plotQuantileDifference(positions, CDF1, CDF2, start, end)
    area = determineArea(positions, CDF1, CDF2, start, end)
    print("Area:", area)


def testCDFCreation():
    # OFF_Bounds = [[0, 2], [8, 10]]
    OFF_Bounds = [[-7.55590262], [-1.9033129]]
    # ON_Bounds = [[3, 6], [9, 10]]
    ON_Bounds = [[-7.53908589], [1.29678683]]
    # quantiles = [0, 0.25, 0.75, 1]
    quantiles = [0, 1]

    positions = determinePositions(OFF_Bounds, ON_Bounds)
    CDF_OFF = determineCDF(positions, OFF_Bounds, quantiles)
    CDF_ON = determineCDF(positions, ON_Bounds, quantiles)

    plotQuantiles(positions, CDF_OFF, CDF_ON, min(positions), max(positions))
    print(positions)
    print(CDF_OFF)
    print(CDF_ON)


def performQuantileBasedScoring():
    # OFF_Bounds = [[np.exp(-5)], [np.exp(-1)]]
    # ON_Bounds = [[np.exp(-4)], [np.exp(-2)]]

    OFF_Bounds = [[np.exp(-6), np.exp(-5), np.exp(-4)], [np.exp(-3), np.exp(-2), np.exp(-1)]]
    ON_Bounds = [[np.exp(-5), np.exp(-4.5), np.exp(-4.25)], [np.exp(-4), np.exp(-3.5), np.exp(0)]]
    OFF_Bounds = np.log(OFF_Bounds)
    ON_Bounds = np.log(ON_Bounds)

    quantiles = [0, 0.2, 0.4, 0.6, 0.8, 1]

    positions = determinePositions(OFF_Bounds, ON_Bounds)
    CDF_OFF = determineCDF(positions, OFF_Bounds, quantiles)
    CDF_ON = determineCDF(positions, ON_Bounds, quantiles)

    # positions = np.log(positions)
    bAsymmetric = True

    if (bAsymmetric):
        start = determineMedian(positions, CDF_OFF)
        end = determineMedian(positions, CDF_ON)
        if (start > end):
            buffer = start
            start = end
            end = buffer
    else:
        start = min(positions)
        end = max(positions)

    plotQuantiles(positions, CDF1=CDF_OFF, CDF2=CDF_ON, start=start, end=end)
    plotQuantileDifference(positions, CDF1=CDF_OFF, CDF2=CDF_ON, start=start, end=end)
    area = determineArea(positions, cdf1=CDF_OFF, cdf2=CDF_ON, start=start, end=end)
    score = np.exp(area)
    print("Area:", area)
    print("Score:", score)


if __name__ == '__main__':
    # testQuantileBasedScoring()

    # testCDFCreation()

    performQuantileBasedScoring()
