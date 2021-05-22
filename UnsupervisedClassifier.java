/**
 * @author Elijah Michaelson
 */

import java.util.*;

/**
  * Run unsupervised clustering algorithms on a corpus of text. News articles from NPR are used that 
  * generally pertain to one of the following topics: COVID-19 vaccination, the Palestine-Israel conflict, 
  * and the GameStop trading frenzy.
  * The text from these websites is extracted as raw html, tokenized, embedded, and used to train a K-Means
  * unsupervised clustering model. Inference is run on the training examples and also on an out-of-sample string.
  * Characterization of the clusters is given by printing the top-weighted terms of each K-Means centroid. 
**/
class UnsupervisedClassifier {

    public static void main(String[] args) {
        String[] urls = {
            "https://www.npr.org/sections/coronavirus-live-updates/2021/05/19/998420256/pfizer-vaccine-can-stay-longer-at-warmer-temperatures-before-being-discarded",
            "https://www.npr.org/2021/05/17/997575473/u-s-to-ship-20-million-additional-covid-vaccine-doses-overseas",
            "https://www.npr.org/2021/05/17/997029362/clinical-trials-underway-for-5-and-younger-covid-19-vaccinations",
            "https://www.npr.org/sections/coronavirus-live-updates/2021/05/07/994839927/pfizer-seeks-full-fda-approval-for-covid-19-vaccine",
            "https://www.npr.org/sections/coronavirus-live-updates/2021/05/04/993519402/pfizer-says-fda-will-soon-authorize-covid-19-vaccine-for-12-15-age-group",
            "https://www.npr.org/sections/coronavirus-live-updates/2021/05/20/998533237/fauci-says-he-expects-vaccines-for-younger-children-by-end-of-year-or-early-2022",
            "https://www.npr.org/sections/health-shots/2021/05/10/994653371/faq-what-you-need-to-know-about-pfizers-covid-vaccine-and-adolescents",
            "https://www.npr.org/2021/05/17/996938155/border-communities-offer-surplus-covid-vaccines-to-canadian-neighbors",
            "https://www.npr.org/2021/05/16/997303657/israeli-airstrikes-on-gaza-continue-after-global-pro-palestinian-protests",
            "https://www.npr.org/2021/04/29/992065009/palestinian-authority-postpones-parliamentary-elections",
            "https://www.npr.org/2021/05/18/997798691/fighting-continues-between-israel-hamas-after-biden-calls-for-cease-fire",
            "https://www.npr.org/2021/05/16/997259390/the-history-behind-tensions-between-israel-palestine",
            "https://www.npr.org/2021/05/19/998336209/why-an-israeli-palestinian-peace-deal-has-lessened-as-a-priority-for-the-u-s",
            "https://www.npr.org/2021/05/14/996760363/what-the-conflict-with-israel-looks-like-to-2-palestinians",
            "https://www.npr.org/2021/05/11/995875422/28-reported-dead-in-gaza-from-israeli-airstrikes-gaza-rockets-kill-2-in-israel",
            "https://www.npr.org/2021/05/08/995014328/conflict-between-israelis-and-palestinians-continue-in-jerusalem",
            "https://www.npr.org/2021/05/12/996079190/latest-violence-quickly-escalates-between-israelis-palestinians",
            "https://www.npr.org/2021/01/31/962479849/reddit-wallstreetbets-founder-calls-gamestop-stock-frenzy-a-symbolic-movement",
            "https://www.npr.org/2021/01/28/961349400/gamestop-how-reddit-traders-occupied-wall-streets-turf",
            "https://www.npr.org/2021/01/28/961722406/the-latest-on-the-gamestop-stock-market-debacle",
            "https://www.npr.org/2021/01/25/960454567/cant-stop-gamestop",
            "https://www.npr.org/2021/01/27/961279048/reddit-users-vs-wall-street-giant-in-fight-over-gamestop-stock-value",
            "https://www.npr.org/2021/01/28/961722436/how-redditors-sent-gamestops-stock-price-through-the-roof",   
            "https://www.npr.org/2021/02/04/964172275/lessons-learned-from-those-who-made-money-and-lost-it-during-the-gamestop-stock-",
            "https://www.npr.org/2021/02/18/969151076/major-players-in-gamestop-stock-saga-appear-before-house-committee",
            "https://www.npr.org/2021/02/03/963361305/will-gamestops-wild-ride-in-the-stock-market-actually-help-its-business",
        };

        int corpusSize = urls.length;
        String[] rawHTML = new String[corpusSize];
        try {
            URLReader reader = new URLReader();

            for (int i = 0; i < corpusSize; i++) {
                rawHTML[i] = reader.get(urls[i]);
            }
        } catch (Exception e) {
            System.out.println("Could not read from provided URL. Exitting.");
            return;
        }
        

        /*
        String[] rawHTML =  {
            "The covid vaccination effort is underway. Pfizer, Moderna, and Janssen all produce covid vaccines. Covid is dangeous and vaccination is recommended.",
            "Vaccines have been shown to be safe and effective. Vaccination is reccomented for all adults. Vaccines are tested for quality. The vaccination program has reduced covid deaths",
            "Israel and Palestine are in dire conflict. Casulaties in both Israel and Palestine have been reported. The conflict has no end in sight.",
            "The conflict between Israel and Palestine has escalated. Civilian casualties in Gaza have been reported.",
            "The GameStop trading frenzy has WallStreet on edge. The stock has shown extreme volatility. GameStop has no comment.",
            "GameStop stock volatility can be linked back to reddit activity."
        };
        int corpusSize = rawHTML.length;
        */


        //clean and tokenize text
        String[] stopwords = { "npr", "news", "i", "may", "how", "some", "all", "more", "what", "so", "says", "said", "it", "its", "as", "the", "a", "an", "these", "their", "right", "amp", "you", "is", "are", "and", "of", "for", "but", "to", "in", "get", "or", "has", "we", "not", "this", "on", "there", "he","have", "be", "that", "can", "from", "function", "https", "parentNode", "script", "inskeep", "music", "soundbite", "were"};
        AlphaOnlyTokenizer tokenizer = new AlphaOnlyTokenizer(stopwords);
        String[][] cleanedTokens = new String[corpusSize][];
        for (int i = 0; i < corpusSize; i++) {
            cleanedTokens[i] = tokenizer.clean(rawHTML[i]);
        }

        //vectorize
        int minOccurence = 10;
        TFIDFVectorizer embedder = new TFIDFVectorizer(cleanedTokens, minOccurence);
        double[][] embeddings = new double[corpusSize][];
        for (int i = 0; i < corpusSize; i++) {
            embeddings[i] = embedder.embed(cleanedTokens[i]);
        }

        //train unsupervised clusterer
        int k = 3;
        int iterations = 500;
        int embedDimension = embedder.getDimension();
        KMeans KM = new KMeans(k, iterations, embedDimension);
        KM.fit(embeddings);

        // infer on training set
        System.out.println("corpus cluster membership predictions: ");
        int[] predictions = new int[corpusSize];
        for (int i = 0; i < corpusSize; i++) {
            predictions[i] = KM.infer(embeddings[i]);
            System.out.print(predictions[i] + " ");
        }
        System.out.println();
        System.out.println();

        //print top terms characterizing cluster
        int printTopN = 6;
        double[][] centroids = KM.getCentroids();
        String[] vocab = embedder.getVocab();
        for (int i = 0; i < KM.getK(); i++) {
            System.out.println("centroid " + i);
            TreeMap<Double, String> topTerms = new TreeMap<Double, String>();
      
            for (int j = 0; j < embedder.getDimension(); j++) {
                topTerms.put(centroids[i][j], vocab[j]);
            }

            int iter = 1;
            for (double weight : topTerms.descendingKeySet()) {
                System.out.println("   " + topTerms.get(weight) + " " + weight);
                if (iter == printTopN) {
                    break;
                }
                iter += 1;
            }
            System.out.println();
        }

        //example out-of-sample inference
        String ex = "the covid vaccination effort is underway. pfizer, moderna, and Janssen are all producing covid vaccines";
        double[] testv = embedder.embed(tokenizer.clean(ex));
        System.out.println("predicted out-of-sample label\n   text: " + ex + "\n   cluster: " + KM.infer(testv));
    }

}






