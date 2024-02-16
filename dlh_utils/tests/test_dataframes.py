'''
Pytesting on Dataframes functions.
'''

from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import StructType,StructField,StringType,LongType,DoubleType
import pandas as pd
from chispa import assert_df_equality
import pytest
from dlh_utils.dataframes import explode,drop_columns,select,cut_off,\
    drop_nulls,union_all,rename_columns,prefix_columns,suffix_columns,split,\
    clone_column,substring,filter_window,concat,coalesced,window,literal_column,\
    date_diff,index_select

pytestmark = pytest.mark.usefixtures("spark")

#############################################################################


class TestExplode(object):
    """Test of explode function."""
    def test_expected(self, spark):

        test_df = spark.createDataFrame(
            (pd.DataFrame({"check": ["iagsigajs"], "before1": ["a_b_c"]}))
        ).select("check", "before1")

        intended_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "check": ["iagsigajs", "iagsigajs", "iagsigajs"],
                        "before1": ["b", "c", "a"],
                    }
                )
            )
        ).select("check", "before1")

        result_df = explode(test_df, "before1", "_")
        assert_df_equality(intended_df, result_df,ignore_row_order=True)


#############################################################################


class TestConcat(object):
    """Test of concat function"""
    def test_expected(self, spark):

        test_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "firstname": [None, "Claire", "Josh", "Bob"],
                        "middlename": ["Maria", None, "", "Greg"],
                        "lastname": ["Jones", None, "Smith", "Evans"],
                        "numeric": [1, 2, None, 4],
                        "after": [
                            "Maria_Jones",
                            "Claire",
                            "Josh_Smith",
                            "Bob_Greg_Evans",
                        ],
                    }
                )
            )
        )

        # Pandas replaces None with NaN in a numeric column. Convert back to Null:
        test_df = test_df.replace(float('nan'), None)

        intended_schema = StructType(
            [
                StructField("firstname", StringType(), True),
                StructField("middlename", StringType(), True),
                StructField("lastname", StringType(), True),
                StructField("numeric", DoubleType(), True),
                StructField("after", StringType(), True),
                StructField("fullname", StringType(), True),
            ]
        )
        intended_data = [
            [None, "Maria", "Jones", 1.0, "Maria_Jones", "Maria_Jones"],
            ["Claire", None, None, 2.0, "Claire", "Claire"],
            ["Josh", "", "Smith", None, "Josh_Smith", "Josh_Smith"],
            ["Bob", "Greg", "Evans", 4.0, "Bob_Greg_Evans", "Bob_Greg_Evans"],
        ]

        intended_df = spark.createDataFrame(intended_data, intended_schema)

        result_df = concat(
            test_df,
            "fullname",
            sep="_",
            columns=["firstname", "middlename", "lastname"],
        )

        assert_df_equality(intended_df, result_df, ignore_row_order=True, ignore_column_order=True)

##############################################################################


class TestDropColumns(object):
    """Test of drop_columns function"""
    def test_expected(self, spark):

        test_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "col1": ["ONE", "TWO", "THREE"],
                        "col2": ["one", "two", "three"],
                        "extra": ["One", "Two", "Three"],
                    }
                )
            )
        )

        intended_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {"col2": ["one", "three", "two"], "extra": ["One", "Three", "Two"]}
                )
            )
        )
        result_df = drop_columns(test_df, subset="col1")
        assert_df_equality(intended_df, result_df, ignore_row_order=True)


##############################################################################


class TestSelect(object):
    def test_expected(self, spark):

        test_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "identifier": [1, 2, 3, 4],
                        "firstName": ["robert", "andrew", "carlos", "john"],
                        "firstLetter": ["r", "a", "c", "j"],
                        "first": ["x", "2", "3", "4"],
                        "numbers": [1, 2, 3, 4],
                    }
                )
            )
        )

        intended_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "firstName": ["robert", "andrew", "john", "carlos"],
                        "firstLetter": ["r", "a", "j", "c"],
                        "first": ["x", "2", "4", "3"],
                    }
                )
            )
        )
        result_df = select(test_df, startswith="first")
        assert_df_equality(intended_df, result_df, ignore_row_order=True)


##########################################################################
class TestCoalesced(object):
    def test_expected(self, spark):

        pdf = pd.DataFrame(
            {
                "extra": [None, None, None, "FO+ UR", None],
                "lower": ["one", None, "one", "four", None],
                "lowerNulls": ["one", "two", None, "four", None],
                "upperNulls": ["ONE", "TWO", None, "FOU  R", None],
                "value": [1, 2, 3, 4, 5],
            }
        )

        pdf = pdf[["extra", "lower", "lowerNulls", "upperNulls", "value"]]
        test_df = spark.createDataFrame((pdf))

        intended_schema = StructType(
            [
                StructField("extra", StringType(), True),
                StructField("lower", StringType(), True),
                StructField("lowerNulls", StringType(), True),
                StructField("upperNulls", StringType(), True),
                StructField("value", LongType(), True),
                StructField("coalesced_col", StringType(), True),
            ]
        )
        intended_data = [
            [None, "one", "one", "ONE", 1, "one"],
            [None, None, "two", "TWO", 2, "two"],
            [None, "one", None, None, 3, "one"],
            ["FO+ UR", "four", "four", "FOU  R", 4, "FO+ UR"],
            [None, None, None, None, 5, "5"],
        ]
        intended_df = spark.createDataFrame(intended_data, intended_schema)

        result_df = coalesced(test_df)
        assert_df_equality(intended_df, result_df, ignore_row_order=True, ignore_column_order=True)

    def test_expected_with_drop(self, spark):
        pdf = pd.DataFrame(
            {
                "lower": ["one", None, "one", "four", None],
                "value": [1, 2, 3, 4, 5],
                "extra": [None, None, None, "FO+ UR", None],
                "lowerNulls": ["one", "two", None, "four", None],
                "upperNulls": ["ONE", "TWO", None, "FOU  R", None],
            }
        )
        pdf = pdf[["lower", "value", "extra", "lowerNulls", "upperNulls"]]

        test_df2 = spark.createDataFrame((pdf))

        intended_schema2 = StructType(
            [
                StructField("coalesced_col", StringType(), True),
            ]
        )
        intended_data2 = [
            ["one"],
            ["2"],
            ["one"],
            ["four"],
            ["5"]
        ]
        intended_df2 = spark.createDataFrame(intended_data2, intended_schema2)

        result_df2 = coalesced(test_df2, drop=True)
        assert_df_equality(intended_df2, result_df2, ignore_row_order=True,
                           ignore_column_order=True)


#################################################################

class TestCutOff(object):
    def test_expected(self, spark):

        test_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {"strings": ["1", "2", "3", "4", "5"], "ints": [1, 2, 3, 4, 5]}
                )
            )
        )

        intended_df = spark.createDataFrame(
            (pd.DataFrame({"strings": ["3", "4", "5"], "ints": [3, 4, 5]}))
        )

        # cutOff does not remove null values when the val is an Int type
        result_df = cut_off(test_df, threshold_column="ints", val=3, mode=">=")
        assert_df_equality(intended_df, result_df, ignore_row_order=True)

        test_df_2 = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "col1": [
                            None,
                            "15-05-1996",
                            "16-04-1996",
                            "17-06-1996",
                            "18-05-1997",
                        ]
                    }
                )
            )
        ).withColumn("col1", F.to_date("col1", "dd-MM-yyyy"))

        intended_df_2 = spark.createDataFrame(
            (pd.DataFrame({"col1": ["18-05-1997"]}))
        ).withColumn("col1", F.to_date("col1", "dd-MM-yyyy"))

        result_df_2 = cut_off(test_df_2, "col1", "1997-01-15", ">=")
        assert_df_equality(intended_df_2, result_df_2, ignore_row_order=True)

        intended_df_3 = spark.createDataFrame(
            (pd.DataFrame({"strings": ["4", "5"], "ints": [4, 5]}))
        )

        result_df3 = cut_off(test_df, threshold_column="ints", val=3, mode=">")
        assert_df_equality(intended_df_3, result_df3, ignore_row_order=True) 

        intended_df_4 = spark.createDataFrame(
            (pd.DataFrame({"strings": ["1", "2", "3"], "ints": [1, 2, 3]}))
        )

        result_df4 = cut_off(test_df, threshold_column="ints", val=4, mode="<")
        assert_df_equality(intended_df_4, result_df4, ignore_row_order=True)

        intended_df_5 = spark.createDataFrame(
            (pd.DataFrame({"strings": ["1", "2", "3"], "ints": [1, 2, 3]}))
        )

        result_df5 = cut_off(test_df, threshold_column="ints", val=3, mode="<=")
        assert_df_equality(intended_df_5, result_df5, ignore_row_order=True)


####################################################################


class TestLiteralColumn(object):
    def test_expected(self, spark):

        test_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "col1": ["one", None, "one", "four", None],
                        "col2": [1, 2, 3, 4, 5],
                    }
                )
            )
        )

        intended_schema = StructType(
            [
                StructField("col1", StringType(), True),
                StructField("col2", LongType(), True),
                StructField("newStr", StringType(), False),
            ]
        )

        intended_data = [
            ["one", 1, "yes"],
            [None, 2, "yes"],
            ["one", 3, "yes"],
            ["four", 4, "yes"],
            [None, 5, "yes"],
        ]
        intended_df = spark.createDataFrame(intended_data, intended_schema)

        result_df = literal_column(test_df, "newStr", "yes")
        assert_df_equality(intended_df, result_df, ignore_row_order=True)


####################################################################


class TestDropNulls(object):
    def test_expected(self, spark):

        test_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "lower": [None, None, "one", "four", "five"],
                        "after": [None, None, "one", "four", "three"],
                    }
                )
            )
        )

        intended_df = spark.createDataFrame(
            (pd.DataFrame({"lower": ["one", "four"], "after": ["one", "four"]}))
        )

        result_df = drop_nulls(test_df, subset="lower", val="five")
        assert_df_equality(intended_df, result_df, ignore_row_order=True)


#####################################################################


class TestUnionAll(object):
    def test_expected(self, spark):

        test_df1 = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "col1": [None, None, "one", "four", "five"],
                        "col2": [None, None, "one", "four", "three"],
                    }
                )
            )
        )

        test_df2 = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "col1": [None, "okay", "dfs", "few", "dfs"],
                        "col2": [None, None, "fdsa", "rew", "trt"],
                    }
                )
            )
        )
        test_df3 = spark.createDataFrame(
            (pd.DataFrame({"col3": [None, "okay", "dfs", "few", "dfs"]}))
        )

        intended_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "col1": [
                            None,
                            None,
                            "one",
                            "four",
                            "five",
                            None,
                            "okay",
                            "dfs",
                            "few",
                            "dfs",
                            "xd",
                            "xd",
                            "xd",
                            "xd",
                            "xd",
                        ],
                        "col2": [
                            None,
                            None,
                            "one",
                            "four",
                            "three",
                            None,
                            None,
                            "fdsa",
                            "rew",
                            "trt",
                            "xd",
                            "xd",
                            "xd",
                            "xd",
                            "xd",
                        ],
                        "col3": [
                            "xd",
                            "xd",
                            "xd",
                            "xd",
                            "xd",
                            "xd",
                            "xd",
                            "xd",
                            "xd",
                            "xd",
                            None,
                            "okay",
                            "dfs",
                            "few",
                            "dfs",
                        ],
                    }
                )
            )
        )

        result_df = union_all(test_df1, test_df2, test_df3, fill="xd")
        assert_df_equality(intended_df, result_df, ignore_row_order=True)


#########################################################################


class TestRenameColumns(object):
    def test_expected(self, spark):

        test_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "col1": [None, None, "one", "four", "five"],
                        "col2": [None, None, "one", "four", "five"],
                    }
                )
            )
        )
        intended_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "first": [None, None, "one", "four", "five"],
                        "second": [None, None, "one", "four", "five"],
                    }
                )
            )
        )

        result_df = rename_columns(
            test_df, rename_dict={"col1": "first", "col2": "second"}
        )
        assert_df_equality(intended_df, result_df, ignore_row_order=True)


#########################################################################


class TestRenameColumns2(object):
    def test_expected(self, spark):

        test_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "abefore": [["a", "b", "c"], None, ["b", "c", "d"]],
                        "bbefore": ["a", None, "b"],
                        "cbefore": ["c", None, "d"],
                    }
                )
            )
        )
        intended_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "aafter": [["a", "b", "c"], None, ["b", "c", "d"]],
                        "bafter": ["a", None, "b"],
                        "cafter": ["c", None, "d"],
                    }
                )
            )
        )
        result_df = rename_columns(
            test_df,
            rename_dict={"abefore": "aafter", "bbefore": "bafter", "cbefore": "cafter"},
        )

        assert_df_equality(intended_df, result_df, ignore_row_order=True)


#########################################################################


class TestPrefixColumns(object):
    def test_expected(self, spark):

        test_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "col1": [None, None, "one", "four", "five"],
                        "col2": [None, None, "one", "four", "five"],
                    }
                )
            )
        )

        intended_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "col1": [None, None, "one", "four", "five"],
                        "mrcol2": [None, None, "one", "four", "five"],
                    }
                )
            )
        )

        result_df = prefix_columns(test_df, prefix="mr", exclude="col1")
        assert_df_equality(intended_df, result_df, ignore_row_order=True)


###########################################################################


class TestSuffixColumns(object):
    def test_expected(self, spark):

        test_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "col1": [None, None, "one", "four", "five"],
                        "col2": [None, None, "one", "four", "five"],
                    }
                )
            )
        )

        intended_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "col1": [None, None, "one", "four", "five"],
                        "col2mr": [None, None, "one", "four", "five"],
                    }
                )
            )
        )
        result_df = suffix_columns(test_df, suffix="mr", exclude="col1")
        assert_df_equality(intended_df, result_df, ignore_row_order=True)


#######################################################################


class TestWindow(object):
    def test_expected(self, spark):

        test_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "col1": ["a", "b", "c", "c", "d", "e", "d"],
                        "col2": [1, 1, 2, 2, 1, 1, 1],
                    }
                )
            )
        )
        intended_schema = StructType(
            [
                StructField("col1", StringType(), True),
                StructField("col2", LongType(), True),
                StructField("new", LongType(), False),
            ]
        )

        intended_data = [
            ["c", 2, 2],
            ["c", 2, 2],
            ["a", 1, 1],
            ["b", 1, 1],
            ["e", 1, 1],
            ["d", 1, 2],
            ["d", 1, 2],
        ]
        intended_df = spark.createDataFrame(intended_data, intended_schema)

        result_df = window(
            test_df, window=["col1", "col2"], target="col2", mode="count", alias="new"
        )
        assert_df_equality(intended_df, result_df, ignore_row_order=True)

        test_df2 = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "col1": ["a", "b", "c", "c", "d", "e", "d"],
                        "col2": [1, 1, 1, 2, 1, 1, 2],
                    }
                )
            )
        )

        intended_schema2 = StructType(
            [
                StructField("col1", StringType(), True),
                StructField("new", LongType(), True),
                StructField("col2", LongType(), True),
            ]
        )
        intended_data2 = [
            ["a", 1, 1],
            ["b", 1, 1],
            ["c", 1, 1],
            ["c", 1, 2],
            ["d", 1, 1],
            ["d", 1, 2],
            ["e", 1, 1],
        ]

        intended_df2 = spark.createDataFrame(intended_data2, intended_schema2)
        result_df2 = window(
            test_df2, window=["col1"], target="col2", mode="min", alias="new"
        ).orderBy("col1", "col2")
        assert_df_equality(intended_df2, result_df2, ignore_row_order=True)

        intended_schema3 = StructType(
            [
                StructField("col1", StringType(), True),
                StructField("new", LongType(), True),
                StructField("col2", LongType(), True),
            ]
        )
        intended_data3 = [
            ["a", 1, 1],
            ["b", 1, 1],
            ["c", 2, 1],
            ["c", 2, 2],
            ["d", 2, 1],
            ["d", 2, 2],
            ["e", 1, 1],
        ]
        intended_df3 = spark.createDataFrame(intended_data3, intended_schema3)
        result_df3 = window(
            test_df2, window=["col1"], target="col2", mode="max", alias="new"
        ).orderBy("col1", "col2")
        assert_df_equality(intended_df3, result_df3, ignore_row_order=True)

        test_df4 = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "col1": [
                            "a",
                            "b",
                            "c",
                            "c",
                            "d",
                            "e",
                            "d",
                            "c",
                            "c",
                            "c",
                            "d",
                            "d",
                        ],
                        "col2": [1, 1, 1, 2, 1, 1, 2, 5, 6, 7, 11, 12],
                    }
                )
            )
        )

        intended_schema4 = StructType(
            [
                StructField("col1", StringType(), True),
                StructField("new", LongType(), True),
                StructField("col2", LongType(), True),
            ]
        )

        intended_data4 = [
            ["a", 1, 1],
            ["b", 1, 1],
            ["c", 7, 1],
            ["c", 7, 2],
            ["c", 7, 5],
            ["c", 7, 6],
            ["c", 7, 7],
            ["d", 12, 1],
            ["d", 12, 2],
            ["d", 12, 11],
            ["d", 12, 12],
            ["e", 1, 1],
        ]

        intended_df4 = spark.createDataFrame(intended_data4, intended_schema4)
        result_df4 = window(
            test_df4, window=["col1"], target="col2", mode="max", alias="new"
        ).orderBy("col1", "col2")
        assert_df_equality(intended_df4, result_df4, ignore_row_order=True)


###############################################################################


class TestSplit(object):
    def test_expected(self, spark):
        test_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {"before": ["a_b_c_d", None], "after": [["a", "b", "c", "d"], None]}
                )
            )
        )

        intended_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "before": ["a_b_c_d", None],
                        "after": [["a", "b", "c", "d"], None],
                        "new": [["a", "b", "c", "d"], None],
                    }
                )
            )
        )
        result_df = split(test_df, "before", col_out="new", split_on="_")
        assert_df_equality(intended_df, result_df, ignore_row_order=True)


###############################################################################


class IndexSelectTesting(object):
    def test_expected(self, spark):
        test_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "before": [["a", "b", "c"], None, ["b", "c", "d"]],
                        "after": ["a", None, "b"],
                        "afterneg": ["c", None, "d"],
                    }
                )
            )
        )

        intended_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "before": [["a", "b", "c"], None, ["b", "c", "d"]],
                        "after": ["a", None, "b"],
                        "afterneg": ["c", None, "d"],
                        "test": ["a", None, "b"],
                    }
                )
            )
        )
        result_df = index_select(test_df, "before", "test", 0)
        assert_df_equality(intended_df, result_df, ignore_row_order=True)


###############################################################################


class TestCloneColumn(object):
    def test_expected(self, spark):

        test_df = spark.createDataFrame(
            (pd.DataFrame({"UPPER": ["ONE£", 'TW""O', "T^^HREE", "FO+UR", "FI@VE"]}))
        )

        intended_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "UPPER": ["ONE£", 'TW""O', "T^^HREE", "FO+UR", "FI@VE"],
                        "NEW": ["ONE£", 'TW""O', "T^^HREE", "FO+UR", "FI@VE"],
                    }
                )
            )
        ).select("UPPER", "NEW")

        result_df = clone_column(test_df, "UPPER", "NEW")
        assert_df_equality(intended_df, result_df, ignore_row_order=True)

    #######################################################################


class TestSubstring(object):
    def test_expected(self, spark):

        test_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "NEW": ["ONE", "TWO", "THREE", "FOUR"],
                        "start": ["ONE", "TWO", "THR", "FOU"],
                        "end": ["ENO", "OWT", "EER", "RUO"],
                    }
                )
            )
        )
        intended_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "NEW": ["ONE", "TWO", "THREE", "FOUR"],
                        "start": ["ONE", "TWO", "THR", "FOU"],
                        "end": ["ENO", "OWT", "EER", "RUO"],
                        "final": ["ONE", "TWO", "THR", "FOU"],
                    }
                )
            )
        ).select("NEW", "end", "start", "final")
        result_df = substring(test_df, "final", "NEW", 1, 3)
        assert_df_equality(intended_df, result_df, ignore_row_order=True, ignore_column_order=True)


##############################################################


class TestFilterWindow(object):
    def test_expected(self, spark):

        test_df1 = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "col1": ["a", "b", "c", "c", "d", "e", "d"],
                        "col2": [1, 1, 2, 2, 1, 1, 1],
                    }
                )
            )
        )

        intended_df1 = spark.createDataFrame(
            (pd.DataFrame({"col1": ["e", "b", "a"], "col2": [1, 1, 1]}))
        )

        result_df1 = filter_window(
            test_df1, "col1", "col2", "count", value=1, condition=True
        )
        assert_df_equality(intended_df1, result_df1, ignore_row_order=True)

        test_df2 = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "col1": ["a", "b", "c", "c", "d", "e", "d"],
                        "col2": [1, 1, 2, 3, 1, 1, 2],
                    }
                )
            )
        )

        intended_df2 = spark.createDataFrame(
            (pd.DataFrame({"col1": ["d", "c"], "col2": [1, 2]}))
        )
        result_df2 = filter_window(test_df2, "col1", "col2", "max", condition=False)

        assert_df_equality(intended_df2, result_df2, ignore_row_order=True)

##############################################################


class TestDateDiff(object):
    def test_expected(self, spark):

        test_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "dob": ['1983-05-12', '1983-03-19', '2012-04-01',
                                '2012-04-01', '2014-05-09','2021-01-12'],
                        "today": ['2023-05-02','2023-05-02','2023-05-02',
                                  '2023-05-02','2023-05-02','2023-05-02'],
                    }
                )
            )
        )

        intended_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "dob": ['1983-05-12', '1983-03-19', '2012-04-01',
                                '2012-04-01', '2014-05-09','2021-01-12'],
                        "today": ['2023-05-02','2023-05-02','2023-05-02',
                                  '2023-05-02','2023-05-02','2023-05-02'],
                        "Difference": [14600.0, 14653.96, 4048.0, 4048.0, 3280.0, 839.96],
                    }
                )
            )
        )

        result_df = date_diff(test_df, 'dob','today',in_date_format='yyyy-MM-dd',units='days')

        assert_df_equality(intended_df, result_df, ignore_row_order=True,
                           ignore_column_order=True)

        intended_df_2 = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "dob": ['1983-05-12', '1983-03-19', '2012-04-01',
                                '2012-04-01', '2014-05-09','2021-01-12'],
                        "today": ['2023-05-02','2023-05-02','2023-05-02',
                                  '2023-05-02','2023-05-02','2023-05-02'],
                        "Difference": [470.97, 472.71, 130.58, 130.58, 105.81, 27.1],
                    }
                )
            )
        )

        result_df2 = date_diff(test_df, 'dob','today',in_date_format='yyyy-MM-dd',units='months')

        assert_df_equality(intended_df_2, result_df2, ignore_row_order=True,
                           ignore_column_order=True)

        intended_df_3 = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "dob": ['1983-05-12', '1983-03-19', '2012-04-01',
                                '2012-04-01', '2014-05-09','2021-01-12'],
                        "today": ['2023-05-02','2023-05-02','2023-05-02',
                                  '2023-05-02','2023-05-02','2023-05-02'],
                        "Difference": [40.0, 40.15, 11.09, 11.09, 8.99, 2.3],
                    }
                )
            )
        )

        result_df3 = date_diff(test_df, 'dob','today',in_date_format='yyyy-MM-dd',units='years')

        assert_df_equality(intended_df_3, result_df3, ignore_row_order=True,
                           ignore_column_order=True)
