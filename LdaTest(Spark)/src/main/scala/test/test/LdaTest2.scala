package test.test



import java.text.BreakIterator

import org.apache.spark.mllib.clustering.{DistributedLDAModel, EMLDAOptimizer, LDA, OnlineLDAOptimizer}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}

/**
  * Created by wenjie on 16-7-21.
  */
object offcialtest {
  private case class Params(
                             input: String = "hdfs://eosdip/user/wenjie19/04_all",
                             k: Int = 800,
                             maxIterations: Int = 3,
                             docConcentration: Double = -1,
                             topicConcentration: Double = -1,
                             vocabSize: Int = 500000,
                             stopwordFile: String = "hdfs://eosdip/user/wenjie19/hgd.txt.dump",
                             algorithm: String = "online",
                             checkpointDir: Option[String] = None,
                             checkpointInterval: Int = 10)

  def main(args: Array[String]) {
    val defaultParams = Params()
    run(defaultParams)
  }

  private def run(params: Params) {
    val conf = new SparkConf().setAppName(s"LDAExample with $params")
    val sc = new SparkContext(conf)
    val preprocessStart = System.nanoTime()
    val (corpus, vocabArray, actualNumTokens) =
      preprocess(sc, params.input, params.vocabSize, params.stopwordFile)
    corpus.cache()
    val actualCorpusSize = corpus.count()
    val actualVocabSize = vocabArray.size
    val preprocessElapsed = (System.nanoTime() - preprocessStart) / 1e9
    println()
    println(s"Corpus summary:")
    println(s"\t Training set size: $actualCorpusSize documents")
    println(s"\t Vocabulary size: $actualVocabSize terms")
    println(s"\t Trainig set size: $actualNumTokens tokens")
    println(s"\t Preprocessing time: $preprocessElapsed sec")
    println()
    // Run LDA.
    val lda = new LDA()
    val optimizer = params.algorithm.toLowerCase match {
      case "em" => new EMLDAOptimizer
      case "online" => new OnlineLDAOptimizer().setMiniBatchFraction(0.05 + 1.0 / actualCorpusSize)
      case _ => throw new IllegalArgumentException(
        s"Only em, online are supported but got ${params.algorithm}.")
    }

    lda.setOptimizer(optimizer)
      .setK(params.k)
      .setMaxIterations(params.maxIterations)
      .setDocConcentration(params.docConcentration)
      .setTopicConcentration(params.topicConcentration)
      .setCheckpointInterval(params.checkpointInterval)
    if (params.checkpointDir.nonEmpty) {
      sc.setCheckpointDir(params.checkpointDir.get)
    }
    val startTime = System.nanoTime()
    val ldaModel = lda.run(corpus)
    val elapsed = (System.nanoTime() - startTime) / 1e9
    println(s"Finished training LDA model.  Summary:")
    println(s"\t Training time: $elapsed sec")

    val topicIndices = ldaModel.describeTopics(maxTermsPerTopic = 50)
    val topics = topicIndices.map { case (terms, termWeights) =>
      terms.zip(termWeights).map { case (term, weight) => (vocabArray(term.toInt), weight) }
    }
    println(topics.size)
    println("----------------------------------")
    println(s"${params.k} topics:")
    topics.zipWithIndex.foreach { case (topic, i) =>
      println(s"TOPIC $i")
      topic.foreach { case (term, weight) =>
        println(s"$term\t$weight")
      }
      println()
    }
    if (ldaModel.isInstanceOf[DistributedLDAModel]) {
      val distLDAModel = ldaModel.asInstanceOf[DistributedLDAModel]
      val avgLogLikelihood = distLDAModel.logLikelihood / actualCorpusSize.toDouble
      println(s"\t likelihood: $avgLogLikelihood")
      println()
    }
    sc.stop()
  }
  /**
    * Load documents, tokenize them, create vocabulary, and prepare documents as term count vectors.
    *
    * @return (corpus, vocabulary as array, total token count in corpus)
    */
  private def preprocess(
                          sc: SparkContext,
                          paths: String,
                          vocabSize: Int,
                          stopwordFile: String): (RDD[(Long, Vector)], Array[String], Long) = {

    val textRDD: RDD[String] = sc.textFile(paths)
    // Split text into words
    val tokenizer = new SimpleTokenizer(sc, stopwordFile)
    val tokenized: RDD[(Long, IndexedSeq[String])] = textRDD.zipWithIndex().map { case (text, id) =>
      id -> tokenizer.getWords(text)
    }
    tokenized.cache()
    val wordCounts: RDD[(String, Long)] = tokenized
      .flatMap { case (_, tokens) => tokens.map(_ -> 1L) }
      .reduceByKey(_ + _)
    wordCounts.cache()
    val fullVocabSize = wordCounts.count()
    val (vocab: Map[String, Int], selectedTokenCount: Long) = {
      val tmpSortedWC: Array[(String, Long)] = if (vocabSize == -1 || fullVocabSize <= vocabSize) {
        wordCounts.collect().sortBy(-_._2)
      } else {
        wordCounts.sortBy(_._2, ascending = false).take(vocabSize)
      }
      (tmpSortedWC.map(_._1).zipWithIndex.toMap, tmpSortedWC.map(_._2).sum)
    }
    val documents = tokenized.map { case (id, tokens) =>
      val wc = new scala.collection.mutable.HashMap[Int, Int]()
      tokens.foreach { term =>
        if (vocab.contains(term)) {
          val termIndex = vocab(term)
          wc(termIndex) = wc.getOrElse(termIndex, 0) + 1
        }
      }
      val indices = wc.keys.toArray.sorted
      val values = indices.map(i => wc(i).toDouble)
      val sb = Vectors.sparse(vocab.size, indices, values)
      (id, sb)
    }
    val vocabArray = new Array[String](vocab.size)
    vocab.foreach { case (term, i) => vocabArray(i) = term }
    (documents, vocabArray, selectedTokenCount)
  }
}

private class SimpleTokenizer(sc: SparkContext, stopwordFile: String) extends Serializable {
  private val stopwords: Set[String] = if (stopwordFile.isEmpty) {
    Set.empty[String]
  } else {
    val stopwordText = sc.textFile(stopwordFile).collect()
    stopwordText.flatMap(_.stripMargin.split("\\s+")).toSet
  }
  // Matches sequences of Unicode letters
  private val allWordRegex = "^(\\p{L}*)$".r
  // Ignore words shorter than this length.
  private val minWordLength = 1
  def getWords(text: String): IndexedSeq[String] = {
    val words = new scala.collection.mutable.ArrayBuffer[String]()
    val wb = BreakIterator.getWordInstance
    wb.setText(text)
    var current = wb.first()
    var end = wb.next()
    while (end != BreakIterator.DONE) {
      // Convert to lowercase
      val word: String = text.substring(current, end).toLowerCase
      word match {
        case allWordRegex(w) if w.length >= minWordLength && !stopwords.contains(w) =>
          words += w
        case _ =>
      }
      current = end
      try {
        end = wb.next()
      } catch {
        case e: Exception =>
          end = BreakIterator.DONE
      }
    }
    words
  }
}
