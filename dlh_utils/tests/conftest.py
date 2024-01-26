"""Import required libraries"""
import pytest
from pyspark.sql import SparkSession
from dlh_utils.sessions import getOrCreateSparkSession

@pytest.fixture(scope="session")
def spark(request):
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

    spark = getOrCreateSparkSession(appName="dlh_utils unit tests", size="small")
    request.addfinalizer(lambda: spark.stop())
    return spark
