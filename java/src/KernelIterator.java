import java.util.Iterator;

/**CLASS: KERNEL ITERATOR
 * Iterates through the greyvalues contained in a kernel
 * An implementation of the following code from RankFilters:
 * for (int kk=0; kk<kernel.length; kk++) {  // y within the cache stripe (we have 2 kernel pointers per cache line)
      for (int p=kernel[kk++]+xCache0; p<=kernel[kk]+xCache0; p++) {
        double v = cache[p];
 */
public class KernelIterator implements Iterator<Float> {
  
  //See class RankFilters
  private float [] cache; //array of greyvalues
  private int x; // x position
  private int [] cachePointers; //array of integer pairs, pointing to the boundary of the kernel
  
  private int kk = 0; //points to a value in cachePointers
  private int p; //points to a value in cache
  
  /**CONSTRUCTOR
   * @param cache array of greyvalues
   * @param x x position
   * @param cachePointers array of integer pairs, pointing to the boundary of the kernel
   */
  public KernelIterator(float[] cache, int x, int[] cachePointers ) {
    this.cache = cache;
    this.x = x;
    this.cachePointers = cachePointers;
    //initalise the 2 loops
    this.p = cachePointers[this.kk] + x;
    this.kk++;
  }
  
  /**IMPLEMENTED: HAS NEXT
   * Return true is the iterator has a value to return
   */
  @Override
  public boolean hasNext() {
    //check if the outer loop has finished
    if (this.kk < this.cachePointers.length) {
      return true;
    } else {
      return false;
    }
  }
  
  /**IMPLEMENTED: NEXT
   * Return the next value in the iterator
   */
  @Override
  public Float next() {
    //get the greyvalue pointed by p in the cache
    Float toBeReturned = new Float(this.cache[p]);
    this.p++; //increment the inner loop variable
    //check if the inner loop has finished
    if (this.p > this.cachePointers[this.kk]+ x) {
      this.kk++; // increment the outer loop variable
      //if the outer loop has not finished, 
      if (this.hasNext()) {
        //initalise the loop varaibles again
        this.p = this.cachePointers[this.kk] + x;
        this.kk++;
      }
    }
    return toBeReturned;
  }
  
}
