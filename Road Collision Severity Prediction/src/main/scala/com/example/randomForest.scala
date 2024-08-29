package com.example

import org.apache.spark.sql.{SparkSession, DataFrame}
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.tuning.{ParamGridBuilder, CrossValidator}

object randomForest {
  def main(args: Array[String]): Unit = {
    // Ensure the correct number of arguments are provided
    if (args.length != 4) {
      println("Usage: randomForest <collision_csv> <parties_csv> <victims_csv> <output_dir>")
      System.exit(1)
    }

    // Initialize Spark session
    val spark = SparkSession.builder()
      .appName("randomForest")
      .getOrCreate()

    // Load the datasets
    val collisionsDF = spark.read.option("header", "true").csv(args(0))
    val partiesDF = spark.read.option("header", "true").csv(args(1))
    val victimsDF = spark.read.option("header", "true").csv(args(2))

    // Filter necessary columns
    val collisionsFiltered = collisionsDF.select("case_id", "collision_severity", "killed_victims", "injured_victims", "weather_1", "alcohol_involved", "primary_collision_factor", "road_surface", "road_condition_1")
    val partiesFiltered = partiesDF.select("case_id", "party_number", "at_fault", "party_sex", "party_age", "party_sobriety", "cellphone_in_use", "party_race")
    val victimsFiltered = victimsDF.select("case_id", "party_number", "victim_degree_of_injury", "victim_sex", "victim_age")

    // Join datasets
    val joinedDF = collisionsFiltered.join(partiesFiltered, "case_id").join(victimsFiltered, Seq("case_id", "party_number"))

    // Remove rows with NULL values
    val joinedDFWithoutNulls = joinedDF.na.drop()

    // List of categorical columns to index
    val categoricalColumns = Array(
      "collision_severity", "weather_1", "road_condition_1", "alcohol_involved", "primary_collision_factor",
      "road_surface", "at_fault", "party_sex", "party_sobriety", "cellphone_in_use", "party_race",
      "victim_degree_of_injury", "victim_sex"
    )

    val columnsToConvert = Array("killed_victims", "injured_victims", "party_age", "victim_age")
    var convertedDF = joinedDFWithoutNulls

    columnsToConvert.foreach { col =>
      convertedDF = convertedDF.withColumn(col, convertedDF(col).cast("double"))
    }

    // Drop the 'case_id' and 'party_number' columns and string index the categorical columns
    var indexedDF = convertedDF.drop("case_id", "party_number")
    categoricalColumns.foreach { col =>
      val indexer = new StringIndexer()
        .setInputCol(col)
        .setOutputCol(s"${col}_index")
        .fit(indexedDF)

      // Transform the DataFrame with the indexed column
      val transformedDF = indexer.transform(indexedDF)

      // Drop the original column and rename the new indexed column
      indexedDF = transformedDF.drop(col).withColumnRenamed(s"${col}_index", col)
    }

    // Specify the label column and features
    val labelColumn = "collision_severity" // Update with the indexed label column
    val featureColumns = indexedDF.columns.filterNot(col => col == labelColumn)

    // Assemble features into a single vector
    val assembler = new VectorAssembler()
      .setInputCols(featureColumns)
      .setOutputCol("features")

    val finalData = assembler.transform(indexedDF).select(labelColumn, "features")

    // Split the data into training and test sets
    val Array(trainingData, testData) = finalData.randomSplit(Array(0.8, 0.2))

    // Prepare a DataFrame to hold the evaluation metrics
    import spark.implicits._
    var metricsDF = Seq.empty[(String, Double, Double, Double, Double, String)].toDF("Model", "Accuracy", "F1 Score", "Precision", "Recall", "BestParams")

    // ======================
    // RandomForest Classifier
    // ======================
    val randomForest = new RandomForestClassifier()
      .setLabelCol(labelColumn)
      .setFeaturesCol("features")

    // Set up parameter grid for RandomForest hyperparameter tuning
    val rfParamGrid = new ParamGridBuilder()
      .addGrid(randomForest.numTrees, Array(10, 20, 50))
      .addGrid(randomForest.maxDepth, Array(5, 10, 20))
      .addGrid(randomForest.maxBins, Array(32, 64))
      .build()

    // Set up CrossValidator for RandomForest
    val rfCrossValidator = new CrossValidator()
      .setEstimator(randomForest)
      .setEstimatorParamMaps(rfParamGrid)
      .setEvaluator(new MulticlassClassificationEvaluator().setLabelCol(labelColumn).setPredictionCol("prediction").setMetricName("accuracy"))
      .setNumFolds(3)

    // Train RandomForest model with hyperparameter tuning
    val rfCVModel = rfCrossValidator.fit(trainingData)

    // Get the best RandomForest model's hyperparameters
    val rfBestParams = rfCVModel.getEstimatorParamMaps
      .zip(rfCVModel.avgMetrics)
      .maxBy(_._2)._1.toString()

    // Make predictions with RandomForest
    val rfPredictions = rfCVModel.transform(testData)

    // Collect RandomForest metrics
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol(labelColumn)
      .setPredictionCol("prediction")

    val rfAccuracy = evaluator.setMetricName("accuracy").evaluate(rfPredictions)
    val rfF1 = evaluator.setMetricName("f1").evaluate(rfPredictions)
    val rfPrecision = evaluator.setMetricName("weightedPrecision").evaluate(rfPredictions)
    val rfRecall = evaluator.setMetricName("weightedRecall").evaluate(rfPredictions)

    // Append RandomForest metrics to the DataFrame
    val rfMetricsRow = Seq(("RandomForestClassifier", rfAccuracy, rfF1, rfPrecision, rfRecall, rfBestParams)).toDF("Model", "Accuracy", "F1 Score", "Precision", "Recall", "BestParams")
    metricsDF = metricsDF.union(rfMetricsRow)

    // Save metrics and parameters to the output directory
    metricsDF.write.option("header", "true").csv(args(3) + "/metrics")

    // Stop the Spark session
    spark.stop()
  }
}
