package miniaicoding.wk1.sparkstreaming
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.regression.LinearRegression

object LinearRegression {
  def main(args: Array[String]): Unit = {
    val sparkSxn: SparkSession = SparkSession.builder()
      .master("local[*]")
      .appName("Spark Streaming")
      .getOrCreate()
    sparkSxn.sparkContext.setLogLevel("ERROR")
    // Load training data
    val training = sparkSxn.read.format("libsvm")
      .load("C:\\MiniAICoding-WAC\\Data\\Datasets\\Data.txt")

    // Creating Linear Regression
    val linearRxn = new LinearRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)

    // Trainig Model
    val linearRxnModel = linearRxn.fit(training)
    // Knowing Intercept
    println(linearRxnModel.intercept)
    // Summarize the model over the training set and print out some metrics
    val linearRxnTrainig = linearRxnModel.summary
    println(linearRxnTrainig.totalIterations)
    linearRxnTrainig.residuals.show()
    println(linearRxnTrainig.rootMeanSquaredError)
    //Prediction, create vector and give here -WK2
    //
    //println(linearRxnModel.predict(Array[]))
  }
}