'''
Pytesting on Profiling functions
'''

import pyspark
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql import Row
from pyspark.sql.types import StructType,StructField,StringType,LongType,IntegerType,FloatType,DecimalType,DoubleType
import pandas as pd
import pytest
import chispa
import re
import pandas as pd
from pyspark.sql import Window
from dlh_utils import utilities as ut
from chispa import assert_df_equality

pytestmark = pytest.mark.usefixtures("spark")

#############################################################################

spark = (
    SparkSession.builder.appName("testing")
    .config("spark.executor.memory", "5g")
    .config("spark.yarn.excecutor.memoryOverhead", "2g")
    .getOrCreate()
)

#############################################################################

class Test_df_describe(object):

    #Test 1
    def test_expected(self,spark):

        test_schema = StructType([
            StructField("ID", IntegerType(), True),
            StructField("age", IntegerType(), True),
            StructField("sex", StringType(), True),
            StructField("pc", StringType(), True),
        ])
        test_data = [
            [1, 1, 'Male', 'gu1111'],
            [2,  1, 'Female', 'gu1211'],
            [3, 56, 'Male', 'gu2111'],
        ]

        test_df = spark.createDataFrame(test_data, test_schema)

        result_df = df_describe(test_df,
                                output_mode='spark',
                                approx_distinct = False,
                                rsd = 0.05)

        intended_schema = StructType([
            StructField("variable", StringType(), True),
            StructField("type", StringType(), True),
            StructField("row_count", LongType(), True),
            StructField("distinct", StringType(), True),
            StructField("percent_distinct", DoubleType(), True), 
            StructField("null", LongType(), True),
            StructField("percent_null", DoubleType(), True),
            StructField("not_null", StringType(), True),
            StructField("percent_not_null", DoubleType(), True),
            StructField("empty", StringType(), True),
            StructField("percent_empty", DoubleType(), True),
            StructField("min", StringType(), True),
            StructField("max", StringType(), True),
            StructField("min_l", StringType(), True),
            StructField("max_l", StringType(), True),
            StructField("max_l_before_point", StringType(), True),
            StructField("min_l_before_point", StringType(), True),
            StructField("max_l_after_point", StringType(), True),
            StructField("min_l_after_point", StringType(), True),
        ])

        intended_data = [
            ['ID', 'int', 3, '3', 100.000000, 0, 0.0, '3', 100.0, '0', 0.0, '1', '3', '1', '1', None, None, None, None],
            ['age', 'int', 3, '2', 66.66666666666666, 0, 0.0, '3', 100.0, '0', 0.0, '1', '56', '1', '2', None, None, None, None],
            ['sex', 'string', 3, '2', 66.66666666666666, 0, 0.0, '3', 100.0, '0', 0.0, None, None, '4', '6', None, None, None, None],
            ['pc', 'string', 3, '3', 100.000000, 0, 0.0, '3', 100.0, '0', 0.0, None, None, '6', '6', None, None, None, None],
        ]


        intended_df = spark.createDataFrame(intended_data, schema=intended_schema)

        assert_df_equality(intended_df, result_df)


    #Test 2
    def test_missing_values(self,spark):

        test_schema = StructType([
        StructField("ID", StringType(), True),
        StructField("Forename", StringType(), True),
        StructField("Middlename", StringType(), True),
        StructField("Surname", StringType(), True),
        StructField("DoB", StringType(), True),
        StructField("Sex", StringType(), True),
        StructField("Postcode", StringType(), True),
        ])

        test_data = [(" ", "Homer","Jay","Simpson","1983-05-12",None,"ET74 2SP"),
        ("2","Marge","Juliet","Simpson","1983-03-19","F","EC1N 8UB"),
        ("3","Bart","Jo-Jo",None,20120401,"M",None),
        ("3","Bart","Jo-Jo","Simpson","2012-04-01","M","HU 9DD"),
        ("4","Lisa","Marie","Simpson","2014-05-09","F",None),
        ("5","Maggie",None,"Simpson",None,"F","G42 8AU")]

        test_df = spark.createDataFrame(test_data, test_schema)


        result_df = df_describe(test_df,
                                output_mode='spark',
                                approx_distinct = False,
                                rsd = 0.05)

        intended_schema = StructType([
          StructField("variable", StringType(), True),
          StructField("type", StringType(), True),
          StructField("row_count", LongType(), True),
          StructField("distinct", StringType(), True),
          StructField("percent_distinct", DoubleType(), True), 
          StructField("null", LongType(), True),
          StructField("percent_null", DoubleType(), True),
          StructField("not_null", StringType(), True),
          StructField("percent_not_null", DoubleType(), True),
          StructField("empty", StringType(), True),
          StructField("percent_empty", DoubleType(), True),
          StructField("min", StringType(), True),
          StructField("max", StringType(), True),
          StructField("min_l", StringType(), True),
          StructField("max_l", StringType(), True),
          StructField("max_l_before_point", StringType(), True),
          StructField("min_l_before_point", StringType(), True),
          StructField("max_l_after_point", StringType(), True),
          StructField("min_l_after_point", StringType(), True),
        ])

        intended_data = [
            ['ID', 'string', 6, '5', 83.33333333333334, 0, 0.00, '6', 100.0, '1', 16.666666666666664, None, None, '1', '1', None, None, None, None],
            ['Forename', 'string', 6, '5', 83.33333333333334, 0, 0.00, '6', 100.0, '0', 0.0, None, None, '4', '6', None, None, None, None],
            ['Middlename', 'string', 6, '4', 66.66666666666666, 1, 16.666666666666664, '5', 83.33333333333334, '0', 0.0, None, None, '3', '6', None, None, None, None],
            ['Surname', 'string', 6, '1', 16.666666666666664, 1, 16.666666666666664, '5', 83.33333333333334, '0', 0.0, None, None, '7', '7', None, None, None, None],
            ['DoB', 'string', 6, '5', 83.33333333333334, 1, 16.666666666666664, '5', 83.33333333333334, '0', 0.0, None, None, '8', '10', None, None, None, None],
            ['Sex', 'string', 6, '2', 33.33333333333333, 1, 16.666666666666664, '5', 83.33333333333334, '0', 0.0, None, None, '1', '1', None, None, None, None],
            ['Postcode', 'string', 6, '4', 66.66666666666666, 2, 33.33333333333333, '4', 66.66666666666666, '0', 0.0, None, None, '6', '8', None, None, None, None],
        ]

        intended_df = spark.createDataFrame(intended_data, schema=intended_schema)

        assert_df_equality(intended_df, result_df)


###############################################################