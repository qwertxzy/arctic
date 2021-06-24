package de.tu_darmstadt.rs.synbio.mapping.util;

import org.paukov.combinatorics3.Generator;

import java.util.Iterator;
import java.util.List;

/*
 * Wrapping k-permutation iterator.
 */

public class PermutationIterator<E> implements Iterator {

    private Iterator<List<E>> combinationIterator;
    private Iterator<List<E>> permutationIterator;

    final private List<E> list;
    final private int k;

    public PermutationIterator(List<E> list, int k) {
        this.list = list;
        this.k = k;
        reset();
    }

    public List<E> next() {

        // special case: only 1 element
        if (list.size() == 1) {
            return list;
        }

        // end of k-permutations --> wrap
        if (!permutationIterator.hasNext() && !combinationIterator.hasNext()) {
            reset();
            return next();
        // get next combination, return first new permutation
        } else if (!permutationIterator.hasNext()) {
            permutationIterator = Generator.permutation((combinationIterator.next())).simple().iterator();
            return permutationIterator.next();
        }
        // return next permutation
        return permutationIterator.next();
    }

    public boolean hasNext() {
        // wrapping = always has next element
        return true;
    }

    private void reset() {
        combinationIterator = Generator.combination(list).simple(k).iterator();
        permutationIterator = Generator.permutation(combinationIterator.next()).simple().iterator();
    }

}
