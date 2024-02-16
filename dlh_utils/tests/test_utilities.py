'''
Pytesting on Linkage functions.
'''

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import StructType,StructField,StringType,LongType,FloatType,\
IntegerType, DoubleType, TimestampType
import pandas as pd
from pandas.util.testing import assert_frame_equal
import numpy as np
import pytest
from chispa import assert_df_equality
from dlh_utils.utilities import describe_metrics, value_counts, \
regex_match, pandas_to_spark, search_files, chunk_list
import datetime as dt
import os

pytestmark = pytest.mark.usefixtures("spark")

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

class TestPandasToSpark(object):
    def test_expected(self,spark):
        pandas_df = pd.DataFrame(
            {
                "colDate": ["19000101"],
                "colInt": [1],
                "colBigInt": [1],
                "colFloat": [1.0],
                "colBigFloat": [1.0],
                "colString": ["hello"]
            }
        )
        pandas_df["colDate"] = pandas_df["colDate"].astype("datetime64[ns]")
        pandas_df["colInt"] = pandas_df["colInt"].astype("int32")
        pandas_df["colBigInt"] = pandas_df["colInt"].astype("int64")
        pandas_df["colFloat"] = pandas_df["colFloat"].astype("float32")
        pandas_df["colBigFloat"] = pandas_df["colBigFloat"].astype("float64")
        result_df = pandas_to_spark(pandas_df)

        intended_schema = StructType(
            [
                StructField("colDate", TimestampType(), True),
                StructField("colInt", IntegerType(), True),
                StructField("colBigInt", LongType(), True),
                StructField("colFloat", FloatType(), True),
                StructField("colBigFloat", DoubleType(), True),
                StructField("colString", StringType(), True),
            ]
        )
        date_val = dt.datetime(1900, 1, 1, 0, 0)
        intended_data = [
            [date_val, 1, 1, 1.0, 1.0, "hello"],
        ]

        intended_df = spark.createDataFrame(intended_data, intended_schema)

        assert_df_equality(result_df, intended_df, ignore_row_order=True)

#############################################################################

class TestSearchFiles(object):
    def test_expected(self, spark):
        path = os.path.dirname(os.path.realpath(__file__))
        result = search_files(path, "import")
        assert sorted(list(result.keys())) == sorted(['test_formatting.py', 'test_linkage.py', 'test_profiling.py', 'test_standardisation.py', 'test_dataframes.py', 'test_flags.py', 'test_utilities.py', 'conftest.py'])

#############################################################################

class TestChunkList(object):
    def test_expected(self, spark):
        data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = chunk_list(data, 4)
        assert result == [
          [0, 1, 2, 3],
          [4, 5, 6, 7],
          [8, 9, 10]
        ]
