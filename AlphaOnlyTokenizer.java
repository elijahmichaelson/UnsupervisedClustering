import java.util.*;

/* 
 * A simple proof-of-concept Tokenizer. 
 * This tokenize extracts alpha-only, lowercase, 1-grams from raw text. Words are split on spaces to 
 * generate token list. Elements from a provided stopword list are removed from tokenization. 
 * Future steps could include noun chunking and stemming. A more extensive stopword list that takes into
 * account the npr source html could be useful as well.
 */
public class AlphaOnlyTokenizer { 
    private String[] stopwordList;

    public AlphaOnlyTokenizer(String[] stopwords) {
        this.stopwordList = stopwords;  
    }

    /**
     * @param text raw text
     * @return token representation with: alpha-only 1-grams, lowercased, no punctuation/numbers/special chars. 
     */
    public String[] clean(String text) {
        text = text.replaceAll("[^a-zA-Z ]", "");  //alpha-only
        text = text.toLowerCase();
        String[] split = text.trim().split("\\s+");

        List<String> cleaned = new ArrayList<String>();  

        for (String word : split) {
            // remove stopwords. 
            boolean isStopword = false;
            for (String stopword : this.stopwordList) {
                if (Objects.equals(stopword.toLowerCase(), word)) {
                    isStopword = true;
                }
            }

            if (isStopword == false) {
                cleaned.add(word);
            }

        }
        return cleaned.toArray(new String[0]);
    }
}