"""Import required libraries"""
import pytest
from pyspark.sql import SparkSession

@pytest.fixture(scope="session")
def spark():
    """Set up the Spark session by using a fixture decorator.

     This will be passed to all the tests by having spark as an
         input. Being able to define the Spark session in this
         way is one advantage of Pytest over unittest.

     Has several options for scope:
         "function" will build and close it for each function
         "session" will last for all the tests

     Generally, "session" will be chosen as the fixture only needs
         to be set up once and this will make the tests run faster
    """

    spark_session = (SparkSession.builder.appName("dlh_utils_tests")
            .config('spark.executor.memory', '5g')
            .config('spark.yarn.excecutor.memoryOverhead', '2g')
            .getOrCreate())

    return spark_session
