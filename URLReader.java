import java.net.*;
import java.io.*;

/* 
 * from: https://docs.oracle.com/javase/tutorial/networking/urls/readingURL.html 
 * Used to fetch raw html from a given url
 */
public class URLReader {
    public URLReader() {
    }

    /**
     * @param url the url to fetch raw html from
     * @return raw html of webpage
     */
    public String get(String url) throws Exception{
        URL oracle = new URL(url);
        BufferedReader in = new BufferedReader(
        new InputStreamReader(oracle.openStream()));

        String inputLine;
        StringBuffer sb = new StringBuffer();

        while ((inputLine = in.readLine()) != null)
            sb.append(inputLine);
        in.close();

        String result = sb.toString().replaceAll("<[^>]*>", "");
        return result;
    }
}