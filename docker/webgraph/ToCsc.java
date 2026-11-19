// Streams a WebGraph BVGraph into gnnsample's binary CSC (data/prepare.py layout):
//   indptr.int64.bin  n+1 little-endian int64 offsets
//   indices.int32.bin little-endian int32 neighbour ids
// Run on the publisher's *transposed* graph: its successor lists are the original graph's
// in-neighbours, i.e. exactly the CSC rows, already in node order. Nothing is sorted and
// memory stays small. Self loops are dropped (prepare.py convention); BVGraph successor
// lists are already sorted and duplicate-free.
import it.unimi.dsi.webgraph.BVGraph;
import it.unimi.dsi.webgraph.ImmutableGraph;
import it.unimi.dsi.webgraph.NodeIterator;
import java.io.BufferedOutputStream;
import java.io.FileOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class ToCsc {
  public static void main(String[] args) throws Exception {
    final String basename = args[0], out = args[1];
    final ImmutableGraph g = BVGraph.loadOffline(basename);  // sequential access only
    final long n = g.numNodes();
    try (BufferedOutputStream ip = new BufferedOutputStream(new FileOutputStream(out + "/indptr.int64.bin"), 1 << 22);
         BufferedOutputStream ix = new BufferedOutputStream(new FileOutputStream(out + "/indices.int32.bin"), 1 << 24)) {
      final ByteBuffer b8 = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN);
      final ByteBuffer b4 = ByteBuffer.allocate(1 << 20).order(ByteOrder.LITTLE_ENDIAN);
      long edges = 0, self = 0, maxDeg = 0, zero = 0;
      b8.clear(); b8.putLong(0); ip.write(b8.array());
      final NodeIterator it = g.nodeIterator();
      for (long v = 0; v < n; v++) {
        final int node = it.nextInt();
        final int d = it.outdegree();
        final int[] succ = it.successorArray();
        long kept = 0;
        for (int j = 0; j < d; j++) {
          if (succ[j] == node) { self++; continue; }
          if (b4.remaining() < 4) { ix.write(b4.array(), 0, b4.position()); b4.clear(); }
          b4.putInt(succ[j]);
          kept++;
        }
        edges += kept;
        maxDeg = Math.max(maxDeg, kept);
        if (kept == 0) zero++;
        b8.clear(); b8.putLong(edges); ip.write(b8.array());
        if ((v & ((1 << 22) - 1)) == 0)
          System.out.printf("node %,d / %,d, edges %,d%n", v, n, edges);
      }
      ix.write(b4.array(), 0, b4.position());
      System.out.printf("DONE nodes=%d edges=%d self_loops_dropped=%d max_in_degree=%d zero_in_degree=%d%n",
                        n, edges, self, maxDeg, zero);
    }
  }
}
