import java.util.*;

/**
 * KMeans implementation.
 * Initialization of centroids is done by randomly selecting k vectors from training set. 
 * this guarantees each initialized centroid falls within region of input space that training examples inhabit.
 * Future work could add random noise to these initializations.
 */
public class KMeans {
    private int k;
    private int trainingIterations;
    private double[][] centroids;
    private int embeddingDimension;   

    public KMeans(int k, int trainingIterations, int embeddingDimension) {
        this.k = k;
        this.trainingIterations = trainingIterations;
        this.embeddingDimension = embeddingDimension;
        this.centroids = new double[k][embeddingDimension];
    }

    /**
     * this function fits the KMeans clustering model
     * @param vectors an array of vector representations
     * @return 
     */
    public void fit(double[][] vectors) {
        //initialize centroids to random training examples
        for (int i = 0; i < this.k; i++) {
            Random random = new Random(); 
            int randomInt = random.nextInt(vectors.length);
            this.centroids[i] = vectors[randomInt];
        }

        for (int i = 0; i < this.trainingIterations; i++) {
            List<double[]>[] groups = new List[this.k];
            for (int j = 0; j < this.k; j++) {
                groups[j] = new ArrayList<double[]>();
            }

            //for each v in vectors, assign v to grouping of closest centroid
            for (double[] vector : vectors) {
                int centroidIndex = infer(vector);
                groups[centroidIndex].add(vector);
            }

            for (int j = 0; j < this.k; j++) {
                //update centroids as mean of closest vectors
                double[][] group = groups[j].toArray(new double[groups[j].size()][this.embeddingDimension]); 
                this.centroids[j] = mean(group);
            }
        }
    }

    /**
     * @param vector a vector representation
     * @return index of closest centroid according to squared euclidean distance. this index is bounded by k
     */
    public int infer(double[] vector) {
        double minDistance = -1.0;
        int minIndex = 0;

        for (int i = 0; i < this.k; i++) {
            double squaredEuclidean = 0.0;
            for (int j = 0; j < this.embeddingDimension; j++) {
                double diff = vector[j] - centroids[i][j];
                squaredEuclidean += diff * diff;
            }

            if (minDistance == -1 || squaredEuclidean <= minDistance) {
                minDistance = squaredEuclidean;
                minIndex = i;
            }
        }
        return minIndex;
    }

    public int getK() {
        return this.k;
    }

    public double[][] getCentroids() {
        return this.centroids;
    }

    /**
     * @param group a group of vectors
     * @return a vector that is the row-wise mean of group of vectors
     */
    private double[] mean(double[][] group) {
        //return index-wise mean of set of vectors
        double[] res = new double[this.embeddingDimension];
        for (int i = 0; i < this.embeddingDimension; i++) {
            double rowSum = 0.0;
            for (double[] vector : group) {
                rowSum += vector[i];
            }
            res[i] = rowSum / group.length;  // calculate mean of row
        }
        return res;
    }
}