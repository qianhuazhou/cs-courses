package edu.illinois.cs.debugging;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.math.BigDecimal;
import java.util.*;

import org.jsoup.Jsoup;
import org.jsoup.nodes.Document;
import org.jsoup.nodes.Element;
import org.jsoup.parser.Parser;
import org.jsoup.select.Elements;

/**
 * Implementation for Spectrum-based Fault Localization (SBFL).
 *
 */
public class SBFL
{

    /**
     * Use Jsoup to parse the coverage file in the XML format.
     *
     * @param file
     * @return a map from each test to the set of lines that it covers
     * @throws FileNotFoundException
     * @throws IOException
     */
    protected static Map<String, Set<String>> readXMLCov(File file) throws FileNotFoundException, IOException {
        FileInputStream fileInputStream = new FileInputStream(file);

        Map<String, Set<String>> res = new HashMap<String, Set<String>>();
        Document document = Jsoup.parse(fileInputStream, "UTF-8", "",
                Parser.xmlParser());

        Elements tests = document.select("test");
        for (Element test : tests) {
            Element name = test.child(0);
            Element covLines = test.child(1);

            Set<String> lines = new HashSet<String>();
            String[] items = covLines.text().split(", ");
            Collections.addAll(lines, items);
            res.put(name.text(), lines);
        }
        return res;
    }

    /**
     * Compute the suspiciousness values for all covered statements based on
     * Ochiai
     *
     * @param cov
     * @param failedTests
     * @return a map from each line to its suspiciousness value
     */

    public static Map<String, Double> Ochiai(Map<String, Set<String>> cov, Set<String> failedTests) {
        Map<String, Double> susp = new LinkedHashMap<>();
        Map<String, Integer> failCount = new HashMap<>();
        Map<String, Integer> passCount = new HashMap<>();
        int total = failedTests.size();
        for (Map.Entry<String, Set<String>> entry : cov.entrySet()) {
            String testName = entry.getKey();
            Set<String> lines = entry.getValue();
            boolean isFailed = failedTests.contains(testName);
            for (String line : lines) {
                if (isFailed) {
                    failCount.put(line, failCount.getOrDefault(line, 0) + 1);
                } else {
                    passCount.put(line, passCount.getOrDefault(line, 0) + 1);
                }
            }
        }
        Set<String> all = new HashSet<>();
        all.addAll(failCount.keySet());
        all.addAll(passCount.keySet());
        for (String line : all) {
            int fail = failCount.getOrDefault(line, 0);
            int pass = passCount.getOrDefault(line, 0);
            double denom = Math.sqrt(total * (fail + pass));
            double score = (denom == 0) ? 0.0 : fail / denom;
            susp.put(line, score);
        }
        return susp;
    }



    /**
     * Get the suspiciousness value for the buggy line from the suspicious
     * statement list
     *
     * @param susp
     * @param buggyLine
     * @return the suspiciousness value for the buggy line
     */
    protected static double getSusp(Map<String, Double> susp,
                                    String buggyLine) {
        // TODO
        return susp.getOrDefault(buggyLine, 0.0);
    }

    /**
     * Rank all statements based on the descending order of their suspiciousness
     * values. Get the rank (print the lowest rank for the tied cases) for the
     * buggy line from the suspicious statement list
     *
     * @param susp
     * @param buggyLine
     * @return the rank of the buggy line
     */
    protected static int getRank(Map<String, Double> susp, String buggyLine) {
        List<Map.Entry<String, Double>> list = new LinkedList<Map.Entry<String, Double>>(
                susp.entrySet());
        // TODO
        Collections.sort(list, (a, b) -> b.getValue().compareTo(a.getValue()));

        HashMap<String, Double> sorted = new LinkedHashMap<String, Double>();
        for (Map.Entry<String, Double> temp: list)
            sorted.put(temp.getKey(), temp.getValue());
        list = new LinkedList<Map.Entry<String, Double>>(sorted.entrySet());
        double line = getSusp(susp, buggyLine);
        int count = 0;
        int i = 0;
        while (i < list.size()) {
            if (list.get(i).getValue() >= line) {
                count++;
                i++;
            } else {
                i++;
            }
        }


        return count;
    }




}
