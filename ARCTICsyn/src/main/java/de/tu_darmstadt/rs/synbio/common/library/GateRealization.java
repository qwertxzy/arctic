package de.tu_darmstadt.rs.synbio.common.library;

import de.tu_darmstadt.rs.synbio.common.LogicType;

public class GateRealization {

    private final String identifier;
    private final LogicType logicType;
    private final String group;
    private final String altIdenfifier;

    private boolean isCharacterized = false;
    private GateCharacterization characterization;

    /* constructor for un-characterized realizations */

    public GateRealization(String identifier, LogicType type, String group, String altIdenfifier) {
        this.identifier = identifier;
        this.logicType = type;
        this.group = group;
        this.altIdenfifier = altIdenfifier;
    }

    /* constructor for characterized realizations */

    public GateRealization(String identifier, LogicType type, String group, String altIdentifier, GateCharacterization characterization) {
        this(identifier, type, group, altIdentifier);
        this.characterization = characterization;
        this.isCharacterized = true;
    }

    public String getIdentifier() {
        return identifier;
    }

    public String getAltIdenfifier() {
        return altIdenfifier;
    }

    public LogicType getLogicType() {
        return logicType;
    }

    public String getGroup() {
        return group;
    }

    public GateCharacterization getCharacterization() {
        return characterization;
    }

    public boolean isCharacterized() {
        return isCharacterized;
    }

    public static class GateCharacterization {

        /* library values */
        private final double ymax;
        private final double ymin;
        private final double k;
        private final double n;

        /* derived values */
        private final double xm;
        private final double ym;
        private final double grad;

        public GateCharacterization(double ymax, double ymin, double k, double n) {

            this.ymax = ymax;
            this.ymin = ymin;
            this.k = k;
            this.n = n;

            this.ym = ((ymax - ymin) / 2 ) + ymin;
            this.xm = Math.pow(((ymax-ymin)/(ym-ymin) - 1), 1/n) * k;
            this.grad = ((ymin - ymax) * n * Math.pow(xm / k, n)) / (xm * Math.pow(1 + Math.pow(xm / k, n), 2));
        }

        /* getters for library values */

        public double getYmax() {
            return ymax;
        }

        public double getYmin() {
            return ymin;
        }

        public double getK() {
            return k;
        }

        public double getN() {
            return n;
        }

        /* getters for derived values */

        public double getXm() {
            return xm;
        }

        public double getYm() {
            return ym;
        }

        public double getGrad() {
            return grad;
        }

        public double getEuclidean(GateRealization.GateCharacterization cmp, Double[] normalization, Double[] proxWeights) {
            return Math.sqrt(proxWeights[0] * Math.pow((this.xm - cmp.getXm()) / normalization[0], 2) + proxWeights[1] * Math.pow((this.ym - cmp.getYm()) / normalization[1], 2) + proxWeights[2] * Math.pow((this.grad - cmp.getGrad()) / normalization[2], 2));
        }
    }

}
