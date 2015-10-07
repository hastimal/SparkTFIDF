import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.feature.IDF
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkContext, SparkConf}
/**
 * Created by hastimal on 10/7/2015.
 */
//Refence:https://spark.apache.org/docs/1.2.0/mllib-feature-extraction.html
object SparkTFIDF {
  def main(args: Array[String]) {
    System.setProperty("hadoop.home.dir","F:\\winutils")
    //St config for Spark
    val conf = new SparkConf().setAppName("SparkTFIDF").setMaster("local[*]").set("spark.executor.memory", "4g")
    val sc = new SparkContext(conf)
    // Load documents (one per line).
    val documents: RDD[Seq[String]] = sc.textFile("src/main/resources/inputData").map(line => line.split(" ").toSeq)
//*While applying HashingTF only needs a single pass to the data, applying IDF needs two passes: first to
//compute the IDF vector and second to scale the term frequencies by IDF.*/
    val hashingTF = new HashingTF()
    val tf = hashingTF.transform(documents)

//*MLlib�s IDF implementation provides an option for ignoring terms which occur in less than a minimum number of documents. In such cases, the IDF for these terms is set to 0.
// This feature can be used by passing the minDocFreq value to the IDF constructor.*/
    tf.cache()
    val idf = new IDF(minDocFreq = 1).fit(tf)//: RDD[Vector]

    val tfidf = idf.transform(tf)//: RDD[Vector]

    tfidf.saveAsTextFile("src/main/resources/outputData")
    println("Successfully done!!")
    sc.stop()
  }
}
