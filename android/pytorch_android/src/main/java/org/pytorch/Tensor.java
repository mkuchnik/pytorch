package org.pytorch;

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.Arrays;
import java.util.Locale;

/**
 * Representation of Tensor, dimensions are stored in {@link Tensor#dims}, elements are stored as
 * {@link java.nio.DirectByteBuffer} of one of the supported types: byte, int, float.
 */
public abstract class Tensor {
  private static final int TYPE_CODE_BYTE = 1;
  private static final int TYPE_CODE_INT32 = 2;
  private static final int TYPE_CODE_FLOAT32 = 3;

  private static final String ERROR_MSG_DATA_BUFFER_NOT_NULL = "Data buffer must be not null";
  private static final String ERROR_MSG_DATA_ARRAY_NOT_NULL = "Data array must be not null";
  private static final String ERROR_MSG_DIMS_NOT_NULL = "Dims must be not null";
  private static final String ERROR_MSG_DIMS_NOT_EMPTY = "Dims must be not empty";
  private static final String ERROR_MSG_INDEX_NOT_NULL = "Index must be not null";
  private static final String ERROR_MSG_DIMS_NON_NEGATIVE = "Dims must be non negative";
  private static final String ERROR_MSG_DATA_BUFFER_MUST_HAVE_NATIVE_BYTE_ORDER =
      "Data buffer must have native byte order (java.nio.ByteOrder#nativeOrder)";
  private static final String ERROR_MSG_DATA_BUFFER_MUST_BE_DIRECT =
      "Data buffer must be direct (java.nio.ByteBuffer#allocateDirect)";

  public final long[] dims;

  private static final int FLOAT_SIZE_BYTES = 4;
  private static final int INT_SIZE_BYTES = 4;

  /**
   * Allocates new direct {@link java.nio.FloatBuffer} with native byte order with specified
   * capacity
   * that can be used to in {@link Tensor#newFloatTensor(long[], FloatBuffer)}.
   *
   * @param numElements capacity (number of elements) of result buffer.
   */
  public static FloatBuffer allocateFloatBuffer(int numElements) {
    return ByteBuffer.allocateDirect(numElements * FLOAT_SIZE_BYTES)
        .order(ByteOrder.nativeOrder())
        .asFloatBuffer();
  }

  /**
   * Allocates new direct {@link java.nio.IntBuffer} with native byte order with specified capacity
   * that can be used to in {@link Tensor#newFloatTensor(long[], IntBuffer)}.
   *
   * @param numElements capacity (number of elements) of result buffer.
   */
  public static IntBuffer allocateIntBuffer(int numElements) {
    return ByteBuffer.allocateDirect(numElements * INT_SIZE_BYTES)
        .order(ByteOrder.nativeOrder())
        .asIntBuffer();
  }

  /**
   * Allocates new direct {@link java.nio.ByteBuffer} with native byte order with specified capacity
   * that can be used to in {@link Tensor#newFloatTensor(long[], ByteBuffer)}.
   *
   * @param numElements capacity (number of elements) of result buffer.
   */
  public static ByteBuffer allocateByteBuffer(int numElements) {
    return ByteBuffer.allocateDirect(numElements).order(ByteOrder.nativeOrder());
  }

  /**
   * Creates new Tensor instance with elements type float and specified dimensions and data as
   * java float array.
   * Content of that java float array will be copied into newly allocated direct buffer
   * {@link Tensor#allocateFloatBuffer(int)}.
   *
   * @param dims Tensor dimensions
   * @param data Tensor elements.
   */
  public static Tensor newFloatTensor(long[] dims, float[] data) {
    checkArgument(data != null, ERROR_MSG_DATA_ARRAY_NOT_NULL);
    checkArgument(dims != null, ERROR_MSG_DIMS_NOT_NULL);
    checkDims(dims);
    checkDimsAndDataCapacityConsistency(data.length, dims);
    final FloatBuffer floatBuffer = allocateFloatBuffer((int) numElements(dims));
    floatBuffer.put(data);
    return new Tensor_float32(floatBuffer, dims);
  }

  /**
   * Creates new Tensor instance with elements type float and specified dimensions and data as
   * java int array.
   * Content of that java int array will be copied into newly allocated direct buffer
   * {@link Tensor#allocateIntBuffer(int)}.
   *
   * @param dims Tensor dimensions
   * @param data Tensor elements.
   */
  public static Tensor newIntTensor(long[] dims, int[] data) {
    checkArgument(data != null, ERROR_MSG_DATA_ARRAY_NOT_NULL);
    checkArgument(dims != null, ERROR_MSG_DIMS_NOT_NULL);
    checkDims(dims);
    checkDimsAndDataCapacityConsistency(data.length, dims);
    final IntBuffer intBuffer = allocateIntBuffer((int) numElements(dims));
    intBuffer.put(data);
    return new Tensor_int32(intBuffer, dims);
  }

  /**
   * Creates new Tensor instance with elements type byte and specified dimensions and data as
   * java byte array.
   * Content of that java byte array will be copied into newly allocated direct buffer
   * {@link Tensor#allocateByteBuffer(int)}.
   *
   * @param dims Tensor dimensions
   * @param data Tensor elements.
   */
  public static Tensor newByteTensor(long[] dims, byte[] data) {
    checkArgument(data != null, ERROR_MSG_DATA_ARRAY_NOT_NULL);
    checkArgument(dims != null, ERROR_MSG_DIMS_NOT_NULL);
    checkDims(dims);
    checkDimsAndDataCapacityConsistency(data.length, dims);
    final ByteBuffer byteBuffer = allocateByteBuffer((int) numElements(dims));
    byteBuffer.put(data);
    return new Tensor_byte(byteBuffer, dims);
  }

  /**
   * Creates new Tensor instance with elements type float and specified dimensions and data.
   *
   * @param dims Tensor dimensions, must be not negative.
   * @param data Direct buffer with native byte order that contains corresponsing to dimensions
   *             number of elements.
   *             Specified buffer is used directly, could be used to change Tensor data.
   * @return new Tensor object with specified dimensions and data.
   */
  public static Tensor newFloatTensor(long[] dims, FloatBuffer data) {
    checkArgument(data != null, ERROR_MSG_DATA_BUFFER_NOT_NULL);
    checkArgument(dims != null, ERROR_MSG_DIMS_NOT_NULL);
    checkDims(dims);
    checkDimsAndDataCapacityConsistency(data.capacity(), dims);
    checkArgument(data.isDirect(), ERROR_MSG_DATA_BUFFER_MUST_BE_DIRECT);
    checkArgument(
        (data.order() == ByteOrder.nativeOrder()),
        ERROR_MSG_DATA_BUFFER_MUST_HAVE_NATIVE_BYTE_ORDER);
    return new Tensor_float32(data, dims);
  }

  /**
   * Creates new Tensor instance with elements type int and specified dimensions and data.
   *
   * @param dims Tensor dimensions, must be not negative.
   * @param data Direct buffer with native byte order that contains corresponsing to dimensions
   *             number of elements.
   *             Specified buffer is used directly, could be used to change Tensor data.
   * @return new Tensor object with specified dimensions and data.
   */
  public static Tensor newIntTensor(long[] dims, IntBuffer data) {
    checkArgument(data != null, ERROR_MSG_DATA_BUFFER_NOT_NULL);
    checkArgument(dims != null, ERROR_MSG_DIMS_NOT_NULL);
    checkDims(dims);
    checkDimsAndDataCapacityConsistency(data.capacity(), dims);
    checkArgument(data.isDirect(), ERROR_MSG_DATA_BUFFER_MUST_BE_DIRECT);
    checkArgument(
        (data.order() == ByteOrder.nativeOrder()),
        ERROR_MSG_DATA_BUFFER_MUST_HAVE_NATIVE_BYTE_ORDER);
    return new Tensor_int32(data, dims);
  }

  /**
   * Creates new Tensor instance with elements type byte and specified dimensions and data.
   *
   * @param dims Tensor dimensions, must be not negative.
   * @param data Direct buffer with native byte order that contains corresponsing to dimensions
   *             number of elements.
   *             Specified buffer is used directly, could be used to change Tensor data.
   * @return new Tensor object with specified dimensions and data.
   */
  public static Tensor newByteTensor(long[] dims, ByteBuffer data) {
    checkArgument(data != null, ERROR_MSG_DATA_BUFFER_NOT_NULL);
    checkArgument(dims != null, ERROR_MSG_DIMS_NOT_NULL);
    checkDims(dims);
    checkDimsAndDataCapacityConsistency(data.capacity(), dims);
    checkArgument(data.isDirect(), ERROR_MSG_DATA_BUFFER_MUST_BE_DIRECT);
    checkArgument(
        (data.order() == ByteOrder.nativeOrder()),
        ERROR_MSG_DATA_BUFFER_MUST_HAVE_NATIVE_BYTE_ORDER);
    return new Tensor_byte(data, dims);
  }

  private Tensor(long[] dims) {
    checkDims(dims);
    this.dims = Arrays.copyOf(dims, dims.length);
  }

  /**
   * Calculates number of elements in tensor with specified dimensions.
   */
  public static long numElements(long[] dims) {
    checkDims(dims);
    int result = 1;
    for (long dim : dims) {
      result *= dim;
    }
    return result;
  }

  /**
   * Returns newly allocated java byte array that contains copy of tensor data.
   *
   * @throws IllegalStateException if it is called for not byte tensor,
   *                               {@link Tensor#isByteTensor()}
   */
  public byte[] getDataAsByteArray() {
    throw new IllegalStateException(
        "Tensor of type " + getClass().getSimpleName() + " cannot return data as byte array.");
  }

  /**
   * Returns newly allocated java int array that contains copy of tensor data.
   *
   * @throws IllegalStateException if it is called for not int tensor, {@link Tensor#isIntTensor()}
   */
  public int[] getDataAsIntArray() {
    throw new IllegalStateException(
        "Tensor of type " + getClass().getSimpleName() + " cannot return data as int array.");
  }

  /**
   * Returns newly allocated java float array that contains copy of tensor data.
   *
   * @throws IllegalStateException if it is called for not float tensor,
   *                               {@link Tensor#isFloatTensor()}
   */
  public float[] getDataAsFloatArray() {
    throw new IllegalStateException(
        "Tensor of type " + getClass().getSimpleName() + " cannot return data as float array.");
  }

  /**
   * @return true if current Tensor instance elements type is byte.
   */
  public boolean isByteTensor() {
    return TYPE_CODE_BYTE == getTypeCode();
  }

  /**
   * @return true if current Tensor instance elements type is int.
   */
  public boolean isIntTensor() {
    return TYPE_CODE_INT32 == getTypeCode();
  }

  /**
   * @return true if current Tensor instance elements type is float.
   */
  public boolean isFloatTensor() {
    return TYPE_CODE_FLOAT32 == getTypeCode();
  }

  abstract int getTypeCode();

  Buffer getRawDataBuffer() {
    throw new IllegalStateException(
        "Tensor of type " + getClass().getSimpleName() + " cannot " + "return raw data buffer.");
  }

  private static String invalidIndexErrorMessage(int[] index, long dims[]) {
    return String.format(
        Locale.US,
        "Invalid index %s for tensor dimensions %s",
        Arrays.toString(index),
        Arrays.toString(dims));
  }

  static class Tensor_float32 extends Tensor {
    private final FloatBuffer data;

    Tensor_float32(FloatBuffer data, long[] dims) {
      super(dims);
      this.data = data;
    }

    @Override
    public float[] getDataAsFloatArray() {
      data.rewind();
      float[] arr = new float[data.remaining()];
      data.get(arr);
      return arr;
    }

    @Override
    int getTypeCode() {
      return TYPE_CODE_FLOAT32;
    }

    @Override
    Buffer getRawDataBuffer() {
      return data;
    }

    @Override
    public String toString() {
      return String.format(
          "Tensor_float32{dims:%s data:%s}",
          Arrays.toString(dims), Arrays.toString(getDataAsFloatArray()));
    }
  }

  static class Tensor_int32 extends Tensor {
    private final IntBuffer data;

    private Tensor_int32(IntBuffer data, long[] dims) {
      super(dims);
      this.data = data;
    }

    @Override
    int getTypeCode() {
      return TYPE_CODE_INT32;
    }

    @Override
    Buffer getRawDataBuffer() {
      return data;
    }

    @Override
    public int[] getDataAsIntArray() {
      data.rewind();
      int[] arr = new int[data.remaining()];
      data.get(arr);
      return arr;
    }

    @Override
    public String toString() {
      return String.format(
          "Tensor_int32{dims:%s data:%s}",
          Arrays.toString(dims), Arrays.toString(getDataAsIntArray()));
    }
  }

  static class Tensor_byte extends Tensor {
    private final ByteBuffer data;

    private Tensor_byte(ByteBuffer data, long[] dims) {
      super(dims);
      this.data = data;
    }

    @Override
    int getTypeCode() {
      return TYPE_CODE_BYTE;
    }

    @Override
    Buffer getRawDataBuffer() {
      return data;
    }

    @Override
    public byte[] getDataAsByteArray() {
      data.rewind();
      byte[] arr = new byte[data.remaining()];
      data.get(arr);
      return arr;
    }

    @Override
    public String toString() {
      return String.format(
          "Tensor_byte{dims:%s data:%s}",
          Arrays.toString(dims), Arrays.toString(getDataAsByteArray()));
    }
  }

  // region checks
  private static void checkArgument(boolean expression, String errorMessage, Object... args) {
    if (!expression) {
      throw new IllegalArgumentException(String.format(Locale.US, errorMessage, args));
    }
  }

  private static void checkDims(long[] dims) {
    checkArgument(dims != null, ERROR_MSG_DIMS_NOT_NULL);
    checkArgument(dims.length > 0, ERROR_MSG_DIMS_NOT_EMPTY);
    for (int i = 0; i < dims.length; i++) {
      checkArgument(dims[i] >= 0, ERROR_MSG_DIMS_NON_NEGATIVE);
    }
  }

  private static void checkIndex(int[] index, long dims[]) {
    checkArgument(dims != null, ERROR_MSG_INDEX_NOT_NULL);

    if (index.length != dims.length) {
      throw new IllegalArgumentException(invalidIndexErrorMessage(index, dims));
    }

    for (int i = 0; i < index.length; i++) {
      if (index[i] >= dims[i]) {
        throw new IllegalArgumentException(invalidIndexErrorMessage(index, dims));
      }
    }
  }

  private static void checkDimsAndDataCapacityConsistency(int dataCapacity, long[] dims) {
    final long numElements = numElements(dims);
    checkArgument(
        numElements == dataCapacity,
        "Inconsistent data capacity:%d and dims number elements:%d dims:%s",
        dataCapacity,
        numElements,
        Arrays.toString(dims));
  }
  // endregion checks

  // Called from native
  private static Tensor nativeNewTensor(ByteBuffer data, long[] dims, int typeCode) {
    if (TYPE_CODE_FLOAT32 == typeCode) {
      return new Tensor_float32(data.asFloatBuffer(), dims);
    } else if (TYPE_CODE_INT32 == typeCode) {
      return new Tensor_int32(data.asIntBuffer(), dims);
    } else if (TYPE_CODE_BYTE == typeCode) {
      return new Tensor_byte(data, dims);
    }
    throw new IllegalArgumentException("Unknown Tensor typeCode");
  }
}
