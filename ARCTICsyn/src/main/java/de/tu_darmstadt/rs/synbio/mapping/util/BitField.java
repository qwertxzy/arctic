package de.tu_darmstadt.rs.synbio.mapping.util;

// Custom class for managing large amounts of binary values as boolean[] would take up too much space
// and java.util.BitSet does not have a fixed length with leading zeroes

import java.util.Arrays;

@SuppressWarnings("unused")
public class BitField {
  private int[] data;
  private int bitPointer;

  public BitField() {
    this(0);
  }

  public BitField(int length) {
    this.data = new int[(int) Math.ceil(length / 32.00)];
    this.bitPointer = length;
  }

  public BitField(boolean[] bits) {
    this(bits.length);

    for (int i = 0; i < bits.length; i++) {
      this.setBit(i, bits[i]);
    }
  }

  public void append(boolean bit) {
    this.setBit(bitPointer, bit);
  }

  public void append(boolean[] bits) {
    for (boolean bit : bits) {
      this.setBit(bitPointer, bit);
    }
  }

  public void append(BitField other) {
    this.append(other.toBooleanArray());
  }

  public void setBit(int index, boolean value) {
    int dataIndex = index / 32;
    int shiftAmount = index % 32;

    assertCapacity(dataIndex);
    bitPointer = Math.max(bitPointer, index + 1);

    int intToModify = data[dataIndex];
    int maskInt;

    if (value) {
      maskInt = 0x0001 << shiftAmount;
      intToModify = intToModify | maskInt;

    } else {
      maskInt = ~(0x0001 << shiftAmount);
      intToModify = intToModify & maskInt;
    }
    data[dataIndex] = intToModify;
  }

  public void flipBit(int index) {
    int dataIndex = index / 32;
    int shiftAmount = index % 32;

    assertCapacity(dataIndex);

    int intToModify = data[dataIndex];
    int maskInt = 0x0001 << shiftAmount;

    data[dataIndex] = intToModify ^ maskInt;
  }

  public boolean getBit(int index) {
    int dataIndex = index / 32;
    int shiftAmount = index % 32;

    if (dataIndex >= data.length) {
      return false;
    }

    return (data[dataIndex] & (0x0001 << shiftAmount)) != 0;
  }

  public BitField subfield(int fromIndex, int toIndex) {
    boolean[] bits = new boolean[toIndex - fromIndex];

    for (int i = fromIndex; i < toIndex; i++) {
      bits[i - fromIndex] = this.getBit(i);
    }

    return new BitField(bits);
  }

  public void or(BitField other) {
    for (int i = 0; i < bitPointer; i++) {
      this.setBit(i, this.getBit(i) || other.getBit(i));
    }
  }

  public void and(BitField other) {
    for (int i = 0; i < bitPointer; i++) {
      this.setBit(i, this.getBit(i) && other.getBit(i));
    }
  }

  public boolean[] toBooleanArray() {
    boolean[] retVal = new boolean[bitPointer];

    for (int i = 0; i < bitPointer; i++) {
      retVal[i] = getBit(i);
    }

    return retVal;
  }

  public int length() {
    return this.bitPointer;
  }

  public static BitField parseInt(int value) {
    BitField retVal = new BitField(32);

    for (int i = 0; i < 32; i++) {
      boolean bitValue = ((value & (0x0001 << i)) != 0);
      retVal.setBit(i, bitValue);
    }

    return retVal;
  }

  @Override
  public boolean equals(Object o) {
    if (o == null) {
      return false;
    }

    if (o.getClass() != this.getClass()) {
      return false;
    }

    final BitField other = (BitField) o;

    return Arrays.equals(this.data, other.data);
  }

  @Override
  public String toString() {
    StringBuilder retString = new StringBuilder();

    // reverse order for big-endian number representation
    for (int i = this.length() - 1; i >= 0; i--) {
      retString.append(this.getBit(i) ? '1' : '0');
    }

    return retString.toString();
  }

  private void assertCapacity(int index) {
    // Check if the int list has enough capacity, if not extend it
    if (index >= data.length) {
      int[] newData = new int[index + 1];

      System.arraycopy(data, 0, newData, 0, data.length);

      data = newData;
    }
  }
}
