'''
Pytesting on Linkage functions.
'''

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import StructType,StructField,StringType,LongType,IntegerType, DoubleType
import pandas as pd
from pandas.util.testing import assert_frame_equal
import numpy as np
import pytest
from chispa import assert_df_equality
from dlh_utils.utilities import describe_metrics, value_counts, regex_match

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

class TestDescribeMetrics(object):

    def test_expected(self,spark):
      df = spark.createDataFrame(
          (
              pd.DataFrame(
                  {
                      "colA": ["A", "A", "B", None, "C", "C", "C", None],
                      "colB": [None, 1, 2, 3, 4, 5, 6, 7]
                  }
              )
          )
      )
      result_df = describe_metrics(df, output_mode='pandas')
      intended_df = pd.DataFrame(
          {
              "variable": ["colA", "colB"],
              "type": ["string", "double"],
              "count": [8, 8],
              "distinct": [3, 8],
              "percent_distinct": [37.5, 100],
              "null": [2, 1],
              "percent_null": [25, 12.5],
              "not_null": [6, 7],
              "percent_not_null": [75, 87.5]
          }
      )      
      assert_frame_equal(result_df, intended_df)

#############################################################################

class TestValueCounts(object):

    def test_expected(self,spark):
      df = spark.createDataFrame(
          (
              pd.DataFrame(
                  {
                      "colA": ["A", "A", "B", None, "C", "C", "C", None],
                      "colB": [None, 1, 2, 3, 3, 5, 5, 6]
                  }
              )
          )
      )
      result_df = value_counts(df, limit=6, output_mode='pandas')
      intended_df = pd.DataFrame(
          {
              "colA": ["C", None, "A", "B", "", ""],
              "colA_count": [3, 2, 2, 1, 0, 0],
              "colB": [3.0, 5.0, 1.0, 2.0, 6.0, np.NaN],
              "colB_count": [2, 2, 1, 1, 1, 1]
          }
      )      
      assert_frame_equal(result_df, intended_df)


#############################################################################

class TestRegexMatch(object):

    def test_expected(self,spark):
      df = spark.createDataFrame(
          (
              pd.DataFrame(
                  {
                      "colA": ["abc123_hello", "", None],
                      "colB": [" abc123_hello", "", None],
                      "colC": ["123_hello", "", None],
                      "colD": ["abc_hello", "", None],
                      "colE": ["abc123hello", "", None],
                      "colF": ["abc123", "", None],
                      "colG": ["abc123_hello", "abc123_hello", "abc123_hello"],
                  }
              )
          )
      )
      regex = "^[a-z]*[0-9]+_"
      result = regex_match(df, regex, limit=10000, cut_off=0.0)
      assert result == ['colA', 'colC', 'colG']
      result = regex_match(df, regex, limit=10000, cut_off=0.6)
      assert result == ['colG']

#############################################################################

