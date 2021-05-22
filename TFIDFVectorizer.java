import java.util.*;

/* 
 * An embedder using the TFIDF vectorization method. 
 * The embedder is initialized with a corpus of document, which determine 
 * the vocabulary and inverse document frequency used in subsequent embedddings. The vectorizer has an 'embed' function which
 * leverages the vocab and idf.
 * 
 * Comments: It is possible to normalize the term frequency component of the TFIDF calculation by dividing TF by Document.length. I opted to 
 * not do normalization on this iteration with the following trade-offs in mind: raw counts are preferred if a small 
 * subset of a text discusses a topic common to other documents in corpus, since it wont be dilluted by other non-common keywords. normalizing
 * is preferred if the topics generally have similar set of terms, but the sequence length of the documents are significantly different.
 *
 * Future work includes the incremental improvement of being able to specify a maximum embedding size, which could be determined by top
 * term frequencies accross the corpus.
 */
public class TFIDFVectorizer {
    private String[] vocab;
    private Map<String, Integer> idf;

    /**
     * @param documents an array of tokenized documents
     * @param minOccurence the minimum number of times a term must occur in a document to be included in vectorization
     * @return  
     */
    public TFIDFVectorizer(String[][] documents, int minOccurence) {
        this.idf = new HashMap<String, Integer>();
        for (String[] document : documents) {
            HashMap<String, Integer> tf = getTermFreq(document);
            tf.entrySet().removeIf(entries->entries.getValue() <= minOccurence);  //remove terms that dont occur often
            for (String term : tf.keySet()) {
                //System.out.println(term + " " + tf.get(term));
                if (this.idf.containsKey(term)) {
                    this.idf.put(term, this.idf.get(term) + 1);
                } else {
                    this.idf.put(term, 1);
                }
            }
        }

        this.idf.entrySet().removeIf(entries->entries.getValue() == documents.length); //remove terms that occur in every document
        this.vocab = this.idf.keySet().toArray(new String[0]);
        System.out.println(vocab.length);
    }

    /**
     * @param document a tokenized representation of a document
     * @return a hashmap of terms to their frequency
     */
     private HashMap<String, Integer> getTermFreq(String[] document) {
        HashMap<String, Integer> termFreq = new HashMap<String, Integer>();
        for (String word : document) {
            if (termFreq.containsKey(word)) {
                termFreq.put(word, termFreq.get(word) + 1);
            } else {
                termFreq.put(word, 1);
            }
        }
        return termFreq;
    }

    /**
     * @param document a tokenized representation of a document
     * @return a vectorized representation of the document using TFIDF
     */
    public double[] embed(String[] document) {
        double[] embedding = new double[this.vocab.length];

        HashMap<String, Integer> tf = getTermFreq(document);
        for (int i = 0; i < this.vocab.length; i++) {
            embedding[i] = 0.0;
            String term = this.vocab[i];
            if (tf.containsKey(term)) {
                embedding[i] = tf.get(term) / this.idf.get(term); //TF * IDF 
            } 
        }

        return embedding;
    }

    public int getDimension() {
        return this.vocab.length;
    }

    public String[] getVocab() {
        return this.vocab;
    }
}
