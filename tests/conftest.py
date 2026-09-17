import pytest

from enterprise_ai.spark.session import get_spark_session


@pytest.fixture(scope="session")
def spark():
    """Provide one Spark session for the test suite."""

    spark_session = get_spark_session()
    spark_session.sparkContext.setLogLevel("ERROR")

    yield spark_session

    spark_session.stop()