'''
Pytesting on Linkage functions.
'''

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import StructType,StructField,StringType,LongType,IntegerType, DoubleType
import pandas as pd
import pytest
from chispa import assert_df_equality
import dlh_utils.profiling

pytestmark = pytest.mark.usefixtures("spark")

@pytest.fixture(scope="session")
def spark(request):
    """fixture for creating a spark context
    Args:
        request: pytest.FixtureRequest object
    """
    spark = (
        SparkSession.builder.appName("dataframe_testing")
        .config("spark.executor.memory", "5g")
        .config("spark.yarn.excecutor.memoryOverhead", "2g")
        .getOrCreate()
    )
    request.addfinalizer(lambda: spark.stop())
    return spark

#############################################################################

class TestDfDescribe(object):

    def test_expected(self,spark):
      # TODO: Write the unit test
      assert False

#############################################################################

class TestValueCounts(object):

    def test_expected(self,spark):
      # TODO: Write the unit test
      assert False

#############################################################################

