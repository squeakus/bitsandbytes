package grn.helper;

import java.lang.reflect.Array;

public class ArrayUtils {

  private ArrayUtils() {
    ;
  }

  /**
   * Fits oversized gene array to the correct size
   * 
   */
  public static <T> T[] fitArray(T[] array, int n) {
    if (array.length == 0 || n == 0 || array[0] == null)
        return (T[]) Array.newInstance(array.getClass().getComponentType(), 0);

    T[] tmp = (T[]) Array.newInstance(array.getClass().getComponentType(), n);
    System.arraycopy(array, 0, tmp, 0, n);
    return tmp;
  }

  /** 
   * Checks if the array is full, if so resize to twice the size and return new array.
   * Otherwise, return the same array
   *
   * @param array to be checked
   * @param n size of array
   * @return The original array, or a copy with double the length
   */
  public static <T>  T[] checkAndResizeArray(T[] array, int n) {
    if (array.length == 0 || n == 0)
      return array;

      if (n == array.length) {
        T[] tmp = (T[]) Array.newInstance(array.getClass().getComponentType(), n*2);
        System.arraycopy(array, 0, tmp, 0, n);
        array = tmp;
      }
      return array;
  }

  /** 
   * Resize the array to the new size, copying what data it can.
   *
   * @param array to be resized
   * @param n new size of array
   * @return The original array, resized
   */
  public static <T> T[] resizeArray(T[] array, int size) {
    int n = size > array.length ? array.length : size;
    T[] tmp = (T[]) Array.newInstance(array.getClass().getComponentType(), size);
    System.arraycopy(array, 0, tmp, 0, n);
    return tmp;
  }
}
