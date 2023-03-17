import pytest
from pyspark.sql import SparkSession

@pytest.fixture(scope="session")
def spark(request):
    """ fixture for creating a spark context
    Args:
        request: pytest.FixtureRequest object
    """
    spark = (SparkSession.builder.appName("dlh_utils_tests")
            .config('spark.executor.memory', '5g')
            .config('spark.yarn.excecutor.memoryOverhead', '2g')
            .getOrCreate())
    request.addfinalizer(lambda: spark.stop())
    return spark
