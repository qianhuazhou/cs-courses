import java.io.*;
import java.lang.reflect.Array;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.*;

public class MP1 {
    Random generator;
    String userName;
    String delimiters = " \t,;.?!-:@[](){}_*/";//每个都是
    String[] stopWordsArray = {"i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours",
            "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its",
            "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
            "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having",
            "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while",
            "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before",
            "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again",
            "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each",
            "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
            "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"};

    public MP1(String userName) {
        this.userName = userName;
    }


    public Integer[] getIndexes() throws NoSuchAlgorithmException {
        Integer n = 10000;
        Integer number_of_lines = 50000;
        Integer[] ret = new Integer[n];
        long longSeed = Long.parseLong(this.userName);
        this.generator = new Random(longSeed);//set seed to be longSeed, same seed with generate same sequence of numbers
        for (int i = 0; i < n; i++) {
            ret[i] = generator.nextInt(number_of_lines);//由seed决定生成随机数,0 ~ number_of_lines-1
        }
        return ret;
    }

    public String[] process() throws Exception {
        String[] topItems = new String[20];
        Integer[] indexes = getIndexes();//不需要处理输入文件中的所有行，而是根据这些索引有选择性地读取和处理文件中的标题。

        //TO DO

        // 模拟输入的 Wikipedia 标题文件
        //String fileName = "input.txt";
        //BufferedReader br = new BufferedReader(new FileReader(fileName));
        BufferedReader br = new BufferedReader(new InputStreamReader(System.in));

        List<String> lines = new ArrayList<>();
        String line;

        // 将文件内容逐行读入到内存列表
        while ((line = br.readLine()) != null) {
            lines.add(line);
        }
        br.close();

        // 保存单词频率的映射
        Map<String, Integer> wordFrequency = new HashMap<>();
        Set<String> stopWords = new HashSet<>(Arrays.asList(stopWordsArray));

        // 处理特定索引的标题
        for (Integer index : indexes) {
            if (index >= lines.size()) continue; // 跳过无效索引
            String title = lines.get(index);

            // 分割标题为单词
            StringTokenizer tokenizer = new StringTokenizer(title, delimiters);
            while (tokenizer.hasMoreTokens()) {
                String word = tokenizer.nextToken().toLowerCase().trim();//trim:去空格
                // 过滤空单词和常见单词
                if (word.isEmpty() || stopWords.contains(word)) continue;

                // 更新单词频率
                wordFrequency.put(word, wordFrequency.getOrDefault(word, 0) + 1);
                /*
                wordFrequency.getOrDefault(word, 0)：
                检查哈希表 wordFrequency 中是否已存在键为 word 的记录：
                如果存在，返回该单词对应的值（即当前频率）。
                如果不存在，返回默认值 0
                等效于
                if (wordFrequency.containsKey(word)) {
                    wordFrequency.put(word, wordFrequency.get(word) + 1);
                } else {
                    wordFrequency.put(word, 1);
                }
                 */
            }
        }

        // 对单词按频率和字母顺序排序
        List<Map.Entry<String, Integer>> sortedWords = new ArrayList<>(wordFrequency.entrySet());
        sortedWords.sort((a, b) -> {
            int freqCompare = b.getValue().compareTo(a.getValue()); // 按频率降序
            return freqCompare != 0 ? freqCompare : a.getKey().compareTo(b.getKey()); // 频率相同时按字母升序
        });

        // 取前 20 个单词
        for (int i = 0; i < 20 && i < sortedWords.size(); i++) {
            topItems[i] = sortedWords.get(i).getKey();
        }

        return topItems;
    }


    public static void main(String args[]) throws Exception {
    	if (args.length < 1){
    		System.out.println("missing the argument");
    	}
    	else{
    		String userName = args[0];
	    	MP1 mp = new MP1(userName);
	    	String[] topItems = mp.process();

	        for (String item: topItems){
	            System.out.println(item);
	        }
	    }
	}

}
