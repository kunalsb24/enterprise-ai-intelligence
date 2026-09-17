from dotenv import load_dotenv

load_dotenv()

from pyspark.sql import SparkSession


def get_spark_session() -> SparkSession:
    """Create a local Spark session for enterprise data processing."""

    return (
        SparkSession.builder
        .appName("EnterpriseAI")
        .master("local[*]")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )