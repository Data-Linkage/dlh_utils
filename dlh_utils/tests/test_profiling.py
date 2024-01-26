'''
Pytesting on Linkage functions.
'''

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import StructType,StructField,StringType,LongType,IntegerType, DoubleType
import pandas as pd
import numpy as np
import pytest
from chispa import assert_df_equality
from pandas.util.testing import assert_frame_equal
import dlh_utils.profiling

pytestmark = pytest.mark.usefixtures("spark")

#############################################################################

class TestDfDescribe(object):

    def test_expected(self,spark):
        df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "colA": ["A", "A", "", None, "C", "C", "C", None],
                        "colB": [None, 1, 2, 3, 4.55, 5, 6, 7]
                    }
                )
            )
        )
        result = dlh_utils.profiling.df_describe(
          df, 
          output_mode='pandas', 
          approx_distinct=False, 
          rsd=0.05
        )
        expected =  pd.DataFrame(
                  {
                      "variable": ["colA", "colB"],
                      "type": ["string", "double"],
                      "row_count": [8, 8],
                      "distinct": ["3", "8"],
                      "percent_distinct": [37.5, 100.0],
                      "null": [2, 0],
                      "percent_null": [25.0, 0],
                      "not_null": ["6","8"],
                      "percent_not_null": [75.0, 100.0],
                      "empty": ["1", "0"],
                      "percent_empty": [12.5, 0],
                      "min": [None, "1.0"],
                      "max": [None, "NaN"],
                      "min_l": ["0", "3"],
                      "max_l": ["1", "4"],
                      "max_l_before_point": [None,"1"],
                      "min_l_before_point": [None, "1"],
                      "max_l_after_point": [None, "2"],
                      "min_l_after_point": [None, "1"]
                  }
              )
        assert_frame_equal(result, expected)

#############################################################################

class TestValueCounts(object):

    def test_expected(self,spark):
      # TODO: Write the unit test
      assert False

#############################################################################

