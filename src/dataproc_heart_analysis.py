import sys
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import StringIndexer, VectorAssembler, StandardScaler
from pyspark.ml import Pipeline

# --- Configuration: GCP Environment Setup ---
GCS_FILE_PATH = "gs://cardiopredict-bronzedata-12345/heart_2022_no_nans.csv"
BIGQUERY_OUTPUT_TABLE = "cardio-predict-project.heart_analytics_dataset.processed_heart_data"
GCS_STAGING_BUCKET = "cardiopredict-bronzedata-12345" 

if len(sys.argv) > 1:
    GCS_FILE_PATH = sys.argv[1]
    print(f"Using GCS Path from command line: {GCS_FILE_PATH}")

def main():
    """
    Executes ETL: Ingests from GCS, cleans/transforms data, and sinks to BigQuery.
    """
    # 1. Initialize Dataproc Spark Session
    spark = SparkSession.builder.appName("HeartDataPrepBQ").getOrCreate()
    spark.conf.set("temporaryGcsBucket", GCS_STAGING_BUCKET)
    
    print("Spark Session Initialized. Preparing data lifecycle transition (Bronze -> Silver)...")

    # --- 2. Data Cleaning & Feature Engineering ---
    try:
        df = spark.read.csv(GCS_FILE_PATH, header=True, inferSchema=True)
        
        # Schema Resolution: Cast to Double for Spark MLlib compatibility
        df = df.withColumn("WeightInKilograms", F.col("WeightInKilograms").cast("double"))
        df = df.withColumn("HeightInMeters", F.col("HeightInMeters").cast("double"))

        # Engineering: BMI Calculation & Data Integrity Filtering
        df = df.withColumn("BMI_Calculated", F.col("WeightInKilograms") / (F.col("HeightInMeters") ** 2))
        df = df.filter(F.col("BMI_Calculated").isNotNull() & ~F.isnan(F.col("BMI_Calculated")))

        # Target Encoding: Convert 'Yes'/'No' to Numerical
        df = df.withColumn("HadHeartAttack_Numeric", F.when(F.col("HadHeartAttack") == "Yes", 1).otherwise(0))

        # Identify Columns for ML Pipeline
        categorical_cols = [
            'Sex', 'GeneralHealth', 'LastCheckupTime', 'PhysicalActivities', 'RemovedTeeth',
            'HadAngina', 'HadStroke', 'HadAsthma', 'HadSkinCancer', 'HadCOPD',
            'HadDepressiveDisorder', 'HadKidneyDisease', 'HadArthritis', 'HadDiabetes',
            'DeafOrHardOfHearing', 'BlindOrVisionDifficulty', 'DifficultyConcentrating',
            'DifficultyWalking', 'DifficultyDressingBathing', 'DifficultyErrands',
            'SmokerStatus', 'ECigaretteUsage', 'ChestScan', 'RaceEthnicityCategory',
            'AgeCategory', 'AlcoholDrinkers', 'HIVTesting', 'FluVaxLast12',
            'PneumoVaxEver', 'TetanusLast10Tdap', 'HighRiskLastYear', 'CovidPos'
        ]
        numerical_cols = ['PhysicalHealthDays', 'MentalHealthDays', 'SleepHours', 'BMI_Calculated']
        
        # Drop raw/redundant columns to maintain Silver Layer efficiency
        df = df.drop(*['State', 'HeightInMeters', 'WeightInKilograms', 'BMI', 'HadHeartAttack'])

    except Exception as e:
        print(f"ERROR during processing: {e}")
        spark.stop()
        return

    # --- 3. ML Pipeline Transformations ---
    # Convert categorical strings to indices
    indexers = [StringIndexer(inputCol=c, outputCol=c + "_indexed", handleInvalid="skip") for c in categorical_cols]
    indexed_cols = [c + "_indexed" for c in categorical_cols]

    # Assemble and Scale numerical features
    num_assembler = VectorAssembler(inputCols=numerical_cols, outputCol="num_features_unscaled", handleInvalid="skip")
    num_scaler = StandardScaler(inputCol="num_features_unscaled", outputCol="num_features_scaled", withStd=True, withMean=False)

    # Build and execute the pipeline
    transform_pipeline = Pipeline(stages=indexers + [num_assembler, num_scaler])
    model = transform_pipeline.fit(df)
    processed_df = model.transform(df)

    # --- 4. Sink to BigQuery (Silver Layer Storage) ---
    # Selecting label + indexed categoricals + raw numericals for BigQuery ML readiness
    final_bq_cols = ["HadHeartAttack_Numeric"] + indexed_cols + numerical_cols
    final_df_bq = processed_df.select(*final_bq_cols)

    try:
        final_df_bq.write \
            .format("bigquery") \
            .option("table", BIGQUERY_OUTPUT_TABLE) \
            .mode("overwrite") \
            .save()
        print("\n✅ Successfully wrote Silver Layer data to BigQuery!")

    except Exception as e:
        print(f"ERROR: Could not write to BigQuery: {e}")

    spark.stop()

if __name__ == "__main__":
    main()