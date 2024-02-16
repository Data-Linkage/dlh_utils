'''
Pytesting on Standardisation functions.
'''

from pyspark.sql import SparkSession
from pyspark.sql.types import StructType,StructField,StringType,IntegerType,DoubleType,LongType
import pandas as pd
from chispa import assert_df_equality
import pytest
from dlh_utils.standardisation import cast_type,standardise_white_space,remove_punct,\
    trim,standardise_case,standardise_date,max_hyphen,max_white_space,align_forenames,\
    add_leading_zeros,group_single_characters,clean_hyphens,standardise_null,fill_nulls,\
    replace,clean_forename,clean_surname,reg_replace,age_at


pytestmark = pytest.mark.usefixtures("spark")

#############################################################################


class TestCastType(object):
    def test_expected(self, spark):

        test_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {"before": [None, "2", "3", "4", "5"], "after": [None, 2, 3, 4, 5]}
                )
            )
        )

        intended_schema = StructType(
            [
                StructField("after", StringType(), True),
                StructField("before", StringType(), True),
            ]
        )

        intended_data = [[float("NaN"), None], [2.0, 2], [3.0, 3], [4.0, 4], [5.0, 5]]
        intended_df = spark.createDataFrame(intended_data, intended_schema)

        # check if it is string first
        result_df = cast_type(test_df, subset="after", types="string")
        assert_df_equality(intended_df, result_df, allow_nan_equality=True,
                           ignore_row_order=True, ignore_column_order=True)

        intended_schema = StructType(
            [
                StructField("after", DoubleType(), True),
                StructField("before", IntegerType(), True),
            ]
        )

        intended_data = [[float("NaN"), None], [2.0, 2], [3.0, 3], [4.0, 4], [5.0, 5]]
        intended_df2 = spark.createDataFrame(intended_data, intended_schema)
        # check if columns are the same after various conversions
        result_df2 = cast_type(test_df, subset="before", types="int")
        assert_df_equality(intended_df2, result_df2, allow_nan_equality=True,
                           ignore_row_order=True, ignore_column_order=True)

##############################################################################


class TestStandardiseWhiteSpace(object):
    def test_expected(self, spark):

        test_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "before": [
                            None,
                            "hello  yes",
                            "hello yes",
                            "hello   yes",
                            "hello yes",
                        ],
                        "after": [
                            None,
                            "hello yes",
                            "hello yes",
                            "hello yes",
                            "hello yes",
                        ],
                        "before2": [
                            None,
                            "hello  yes",
                            "hello yes",
                            "hello   yes",
                            "hello yes",
                        ],
                        "after2": [
                            None,
                            "hello_yes",
                            "hello_yes",
                            "hello_yes",
                            "hello_yes",
                        ],
                    }
                )
            )
        )
        intended_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "before": [
                            None,
                            "hello yes",
                            "hello yes",
                            "hello yes",
                            "hello yes",
                        ],
                        "after": [
                            None,
                            "hello yes",
                            "hello yes",
                            "hello yes",
                            "hello yes",
                        ],
                        "before2": [
                            None,
                            "hello  yes",
                            "hello yes",
                            "hello   yes",
                            "hello yes",
                        ],
                        "after2": [
                            None,
                            "hello_yes",
                            "hello_yes",
                            "hello_yes",
                            "hello_yes",
                        ],
                    }
                )
            )
        )
        result_df = standardise_white_space(test_df, subset="before", wsl="one")
        assert_df_equality(intended_df, result_df,
                           ignore_row_order=True, ignore_column_order=True)

        intended_df2 = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "before": [
                            None,
                            "hello  yes",
                            "hello yes",
                            "hello   yes",
                            "hello yes",
                        ],
                        "after": [
                            None,
                            "hello yes",
                            "hello yes",
                            "hello yes",
                            "hello yes",
                        ],
                        "before2": [
                            None,
                            "hello_yes",
                            "hello_yes",
                            "hello_yes",
                            "hello_yes",
                        ],
                        "after2": [
                            None,
                            "hello_yes",
                            "hello_yes",
                            "hello_yes",
                            "hello_yes",
                        ],
                    }
                )
            )
        )
        result_df2 = standardise_white_space(test_df, subset="before2", fill="_")
        assert_df_equality(intended_df2, result_df2,
                           ignore_row_order=True, ignore_column_order=True)

    def test_expected_wsl_none(self, spark):

        test_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "before": [
                            None,
                            "hello  yes",
                            "hello yes",
                            "hello   yes",
                            "hello yes",
                        ],
                        "after": [
                            None,
                            "hello yes",
                            "hello yes",
                            "hello yes",
                            "hello yes",
                        ],
                        "before2": [
                            None,
                            "hello  yes",
                            "hello yes",
                            "hello   yes",
                            "hello yes",
                        ],
                        "after2": [
                            None,
                            "hello_yes",
                            "hello_yes",
                            "hello_yes",
                            "hello_yes",
                        ],
                    }
                )
            )
        )

        intended_df3 = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "before": [
                            None,
                            "hello  yes",
                            "hello yes",
                            "hello   yes",
                            "hello yes",
                        ],
                        "after": [
                            None,
                            "hello yes",
                            "hello yes",
                            "hello yes",
                            "hello yes",
                        ],
                        "before2": [
                            None,
                            "helloyes",
                            "helloyes",
                            "helloyes",
                            "helloyes",
                        ],
                        "after2": [
                            None,
                            "hello_yes",
                            "hello_yes",
                            "hello_yes",
                            "hello_yes",
                        ],
                    }
                )
            )
        )

        result_df3 = standardise_white_space(test_df, subset="before2", wsl = 'none')
        assert_df_equality(intended_df3, result_df3,
                           ignore_row_order=True, ignore_column_order=True)

##############################################################################

class TestAlignForenames(object):
    def test_expected(self, spark):

        test_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "first_name": ["David Joe", "Dan James", "Neil Oliver", "Rich", "Rachel"],
                        "middle_name": [" ", 'Jim', "", "Fred", "Amy"],
                        "id": [101, 102, 103, 104, 105],
                    }
                )
            )
        )

        intended_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "first_name": ["David", "Dan", "Neil", "Rich", "Rachel"],
                        "middle_name": ['Joe', "James Jim", "Oliver", "Fred", "Amy"],
                        "id": [101, 102, 103, 104, 105],
                    }
                )
            )
        )

        result_df = align_forenames(test_df, 'first_name', 'middle_name', 'id', sep=' ')
        assert_df_equality(intended_df, result_df,
                           ignore_row_order=True, ignore_column_order=True)


##############################################################################


class TestRemovePunct(object):
    def test_expected(self, spark):

        test_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "after": ["ONE", "TWO", "THREE", "FOUR", "FI^VE"],
                        "before": [None, 'TW""O', "TH@REE", "FO+UR", "FI@^VE"],
                        "extra": [None, "TWO", "TH@REE", "FO+UR", "FI@^VE"],
                    }
                )
            )
        )

        intended_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "after": ["ONE", "TWO", "THREE", "FOUR", "FI^VE"],
                        "before": [None, "TWO", "THREE", "FOUR", "FI^VE"],
                        "extra": [None, "TWO", "TH@REE", "FO+UR", "FI@^VE"],
                    }
                )
            )
        )

        result_df = remove_punct(test_df, keep="^", subset=["after", "before"])

        assert_df_equality(intended_df, result_df,
                           ignore_row_order=True, ignore_column_order=True)


##############################################################################


class TestTrim(object):
    def test_expected(self, spark):

        test_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "before1": [None, "", " th re e", "  four ", "  f iv  e "],
                        "before2": [None, " ", " th re e", "  four ", "  f iv  e "],
                        "numeric": [1, 2, 3, 4, 5],
                        "after": [None, "", "th re e", "four", "f iv  e"],
                    }
                )
            )
        )

        intended_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "before1": [None, "", "th re e", "four", "f iv  e"],
                        "before2": [None, " ", " th re e", "  four ", "  f iv  e "],
                        "numeric": [1, 2, 3, 4, 5],
                        "after": [None, "", "th re e", "four", "f iv  e"],
                    }
                )
            )
        )

        result_df = trim(test_df, subset=["before1", "numeric", "after"])

        assert_df_equality(intended_df, result_df,
                           ignore_row_order=True, ignore_column_order=True)


##############################################################################


class TestStandardiseCase(object):
    def test_expected(self, spark):

        test_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "upper": ["ONE", "TWO", "THREE"],
                        "lower": ["one", "two", "three"],
                        "title": ["One", "Two", "Three"],
                    }
                )
            )
        )

        intended_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "upper": ["ONE", "TWO", "THREE"],
                        "lower": ["ONE", "TWO", "THREE"],
                        "title": ["One", "Two", "Three"],
                    }
                )
            )
        )

        result_df = standardise_case(test_df, subset="lower", val="upper")
        assert_df_equality(intended_df, result_df,
                           ignore_row_order=True, ignore_column_order=True)

        intended_df2 = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "upper": ["one", "two", "three"],
                        "lower": ["one", "two", "three"],
                        "title": ["One", "Two", "Three"],
                    }
                )
            )
        )

        result_df2 = standardise_case(test_df, subset="upper", val="lower")
        assert_df_equality(intended_df2, result_df2,
                           ignore_row_order=True, ignore_column_order=True)

        intended_df3 = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "upper": ["ONE", "TWO", "THREE"],
                        "lower": ["One", "Two", "Three"],
                        "title": ["One", "Two", "Three"],
                    }
                )
            )
        )

        result_df3 = standardise_case(test_df, subset="lower", val="title")
        assert_df_equality(intended_df3, result_df3,
                           ignore_row_order=True, ignore_column_order=True)    


##############################################################################


class TestStandardiseDate(object):
    def test_expected(self, spark):

        test_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "before": [None, "14-05-1996", "15-04-1996"],
                        "after": [None, "1996-05-14", "1996-04-15"],
                        "slashed": [None, "14/05/1996", "15/04/1996"],
                        "slashedReverse": [None, "1996/05/14", "1996/04/15"],
                    }
                )
            )
        )

        intended_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "before": [None, "1996-05-14", "1996-04-15"],
                        "after": [None, "1996-05-14", "1996-04-15"],
                        "slashed": [None, "14/05/1996", "15/04/1996"],
                        "slashedReverse": [None, "1996/05/14", "1996/04/15"],
                    }
                )
            )
        )

        result_df = standardise_date(test_df, col_name="before")
        assert_df_equality(intended_df, result_df,
                           ignore_row_order=True, ignore_column_order=True)

        intended_df2 = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "before": [None, "14/05/1996", "15/04/1996"],
                        "after": [None, "1996-05-14", "1996-04-15"],
                        "slashed": [None, "14/05/1996", "15/04/1996"],
                        "slashedReverse": [None, "1996/05/14", "1996/04/15"],
                    }
                )
            )
        )

        result_df2 = standardise_date(
            test_df, col_name="before", out_date_format="dd/MM/yyyy"
        )
        assert_df_equality(intended_df2, result_df2,
                           ignore_row_order=True, ignore_column_order=True)

        intended_df3 = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "before": [None, "14-05-1996", "15-04-1996"],
                        "after": [None, "1996-05-14", "1996-04-15"],
                        "slashed": [None, "14/05/1996", "15/04/1996"],
                        "slashedReverse": [None, "14-05-1996", "15-04-1996"],
                    }
                )
            )
        )

        result_df3 = standardise_date(
            test_df,
            col_name="slashedReverse",
            in_date_format="yyyy/MM/dd",
            out_date_format="dd-MM-yyyy",
        )

        assert_df_equality(intended_df3, result_df3,
                           ignore_row_order=True, ignore_column_order=True)

        intended_df4 = spark.createDataFrame(
          (
              pd.DataFrame(
                  {
                      "before": [None, "14/05/1996", "15/04/1996"],
                      "after": [None, "1996-05-14", "1996-04-15"],
                      "slashed": [None, "14/05/1996", "15/04/1996"],
                      "slashedReverse": [None, "1996/05/14", "1996/04/15"],
                  }
              )
          )
        )

        result_df4 = standardise_date(
            test_df,
            col_name="before",
            in_date_format="dd-MM-yyyy",
            out_date_format="dd/MM/yyyy"
        )

        assert_df_equality(intended_df4, result_df4,
                         ignore_row_order=True, ignore_column_order=True)

##############################################################################


class TestMaxHyphen(object):
    def test_expected(self, spark):

        # max hyphen gets rid of any hyphens that does not match
        # or is under the limit
        test_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "before": [
                        "james--brad",
                        "tom----ridley",
                        "chicken-wing",
                        "agent-----john",
                    ],
                    "after2": ["james--brad", "tomridley", "chicken-wing", "agentjohn"],
                    "after4": [
                        "james--brad",
                        "tom----ridley",
                        "chicken-wing",
                        "agentjohn",
                    ],
                }
            )
        )

        intended_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "before": [
                        "james--brad",
                        "tom--ridley",
                        "chicken-wing",
                        "agent--john",
                    ],
                    "after2": ["james--brad", "tomridley", "chicken-wing", "agentjohn"],
                    "after4": [
                        "james--brad",
                        "tom----ridley",
                        "chicken-wing",
                        "agentjohn",
                    ],
                }
            )
        )
        result_df = max_hyphen(test_df, limit=2, subset=["before"])
        assert_df_equality(intended_df, result_df,
                           ignore_row_order=True, ignore_column_order=True)

        intended_df2 = spark.createDataFrame(
            pd.DataFrame(
                {
                    "before": [
                        "james--brad",
                        "tom----ridley",
                        "chicken-wing",
                        "agent----john",
                    ],
                    "after2": ["james--brad", "tomridley", "chicken-wing", "agentjohn"],
                    "after4": [
                        "james--brad",
                        "tom----ridley",
                        "chicken-wing",
                        "agentjohn",
                    ],
                }
            )
        )
        result_df2 = max_hyphen(test_df, limit=4, subset=["before"])
        assert_df_equality(intended_df2, result_df2,
                           ignore_row_order=True, ignore_column_order=True)


##############################################################################


class TestMaxWhiteSpace(object):
    def test_expected(self, spark):

        # max_white_space gets rid of any whitespace that does not match
        # or is under the limit
        test_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "before": [
                        "james  brad",
                        "tom    ridley",
                        "chicken wing",
                        "agent     john",
                    ],
                    "after2": ["james  brad", "tomridley", "chicken wing", "agentjohn"],
                    "after4": [
                        "james  brad",
                        "tom    ridley",
                        "chicken wing",
                        "agentjohn",
                    ],
                }
            )
        )

        intended_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "before": ["james  brad", "tomridley", "chicken wing", "agentjohn"],
                    "after2": ["james  brad", "tomridley", "chicken wing", "agentjohn"],
                    "after4": [
                        "james  brad",
                        "tom    ridley",
                        "chicken wing",
                        "agentjohn",
                    ],
                }
            )
        )

        result_df = max_white_space(test_df, limit=2, subset=["before"])
        assert_df_equality(intended_df, result_df,
                           ignore_row_order=True, ignore_column_order=True)

        intended_df2 = spark.createDataFrame(
            pd.DataFrame(
                {
                    "before": [
                        "james  brad",
                        "tom    ridley",
                        "chicken wing",
                        "agentjohn",
                    ],
                    "after2": ["james  brad", "tomridley", "chicken wing", "agentjohn"],
                    "after4": [
                        "james  brad",
                        "tom    ridley",
                        "chicken wing",
                        "agentjohn",
                    ],
                }
            )
        )

        result_df2 = max_white_space(test_df, limit=4, subset=["before"])
        assert_df_equality(intended_df2, result_df2,
                           ignore_row_order=True, ignore_column_order=True)


##############################################################################


class TestAlignForenames(object):
    def test_expected(self, spark):

        test_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "identifier": [1, 2, 3, 4],
                        "firstName": [
                            "robert green",
                            "andrew",
                            "carlos senior",
                            "john wick",
                        ],
                        "middleName": [None, "hog", None, ""],
                    }
                )
            )
        )

        intended_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "identifier": [1, 2, 3, 4],
                        "firstName": ["robert", "andrew", "carlos", "john"],
                        "middleName": ["green", "hog", "senior", "wick"],
                    }
                )
            )
        )

        result_df = align_forenames(test_df, "firstName", "middleName", "identifier")
        assert_df_equality(intended_df, result_df,
                           ignore_row_order=True, ignore_column_order=True)


##############################################################################


class TestAddLeadingZeros(object):

    def test_expected(self, spark):
        test_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "before1": ["1-2-12", "2-2-12", "3-2-12", "4-2-12", None],
                        "after1": ["01-2-12", "02-2-12", "03-2-12", "04-2-12", None],
                    }
                )
            )
        )

        intended_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "before1": ["01-2-12", "02-2-12", "03-2-12", "04-2-12", None],
                        "after1": ["01-2-12", "02-2-12", "03-2-12", "04-2-12", None],
                    }
                )
            )
        )

        result_df = add_leading_zeros(test_df, subset=["before1"], n=7)
        assert_df_equality(intended_df, result_df,
                           ignore_row_order=True, ignore_column_order=True)


##############################################################################


class TestGroupSingleCharacters(object):
    def test_expected(self, spark):

        test_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "before1": [None, "", "-t-h r e e", "four", "f i v e", "six ",
                                    " seven", "eigh t", "n ine", "t  e    n", "e leve n"],
                        "before2": [None, "", "-t-h r e e", "four", "f i v e", "six ",
                                    " seven", "eigh t", "n ine", "t  e    n", "e leve n"],
                        "after": [None, "", "-t-h ree", "four", "five", "six ",
                                  " seven", "eigh t", "n ine", "ten", "e leve n"],
                    }
                )
            )
        )

        intended_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "before1": [None, "", "-t-h ree", "four", "five", "six ",
                                    " seven", "eigh t", "n ine", "ten", "e leve n"],
                        "before2": [None, "", "-t-h r e e", "four", "f i v e", "six ",
                                    " seven", "eigh t", "n ine", "t  e    n", "e leve n"],
                        "after": [None, "", "-t-h ree", "four", "five", "six ",
                                  " seven", "eigh t", "n ine", "ten", "e leve n"],
                    }
                )
            )
        )
        result_df = group_single_characters(test_df, subset="before1")
        assert_df_equality(intended_df, result_df,
                           ignore_row_order=True, ignore_column_order=True)

    def test_expected_include_terminals(self, spark):

        test_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "before1": [None, "", "-t-h r e e", "four", "f i v e", "six ",
                                    " seven", "eigh t", "n ine", "t  e    n", "e leve n"],
                        "before2": [None, "", "-t-h r e e", "four", "f i v e", "six ",
                                    " seven", "eigh t", "n ine", "t  e    n", "e leve n"],
                        "after": [None, "", "-t-h ree", "four", "five", "six ",
                                  " seven", "eight", "nine", "ten", "eleven"],
                    }
                )
            )
        )

        intended_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "before1": [None, "", "-t-h ree", "four", "five", "six ",
                                    " seven", "eight", "nine", "ten", "eleven"],
                        "before2": [None, "", "-t-h r e e", "four", "f i v e", "six ",
                                    " seven", "eigh t", "n ine", "t  e    n", "e leve n"],
                        "after": [None, "", "-t-h ree", "four", "five", "six ",
                                  " seven", "eight", "nine", "ten", "eleven"],
                    }
                )
            )
        )
        result_df = group_single_characters(
            test_df,
            subset="before1",
            include_terminals=True
        )
        assert_df_equality(intended_df, result_df,
                           ignore_row_order=True, ignore_column_order=True)


##############################################################################


class TestCleanHyphens(object):
    def test_expected(self, spark):

        test_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "before1": [None, "", "th- ree", "--fo - ur", "fi -ve-"],
                        "before2": [None, "", "th- ree", "fo - ur", "fi -ve"],
                        "after": [None, "", "th-ree", "fo-ur", "fi-ve"],
                    }
                )
            )
        )

        intended_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "before1": [None, "", "th-ree", "fo-ur", "fi-ve"],
                        "before2": [None, "", "th- ree", "fo - ur", "fi -ve"],
                        "after": [None, "", "th-ree", "fo-ur", "fi-ve"],
                    }
                )
            )
        )

        result_df = clean_hyphens(test_df, subset="before1")
        assert_df_equality(intended_df, result_df,
                           ignore_row_order=True, ignore_column_order=True)


#############################################################################


class TestStandardiseNull(object):
    def test_expected(self, spark):

        test_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "before1": [None, "", "  ", "-999", "####", "KEEP"],
                        "before2": [None, "", "  ", "-999", "####", "KEEP"],
                        "after": [None, None, None, None, None, "KEEP"],
                    }
                )
            )
        )

        intended_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "before1": [None, None, None, None, None, "KEEP"],
                        "before2": [None, "", "  ", "-999", "####", "KEEP"],
                        "after": [None, None, None, None, None, "KEEP"],
                    }
                )
            )
        )

        result_df = standardise_null(
            test_df, replace="^-[0-9]|^[#]+$|^$|^\\s*$", subset="before1", regex=True
        )
        assert_df_equality(intended_df, result_df)

        intended_df2 = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "before1": [None, "", "  ", None, "####", "KEEP"],
                        "before2": [None, "", "  ", "-999", "####", "KEEP"],
                        "after": [None, None, None, None, None, "KEEP"],
                    }
                )
            )
        )

        result_df2 = standardise_null(
            test_df, replace="-999", subset="before1", regex=False
        )
        assert_df_equality(intended_df2, result_df2,
                           ignore_row_order=True, ignore_column_order=True)


##############################################################################


class TestFillNulls(object):
    def test_expected(self, spark):

        test_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "before": ["abcd", None, "fg", ""],
                        "numeric": [1, 2, None, 3],
                        "after": ["abcd", None, "fg", ""],
                        "afternumeric": [1, 2, 0, 3],
                    }
                )
            )
        )

        intended_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "before": ["abcd", "0", "fg", ""],
                        "numeric": [1.0, 2.0, 0.0, 3.0],
                        "after": ["abcd", "0", "fg", ""],
                        "afternumeric": [1, 2, 0, 3],
                    }
                )
            )
        )

        result_df = fill_nulls(test_df, 0)
        assert_df_equality(intended_df, result_df,
                           ignore_row_order=True, ignore_column_order=True)


##############################################################################

class TestAgeAt(object):
    def test_expected(self, spark):

        test_schema = StructType([
            StructField("ID", LongType(), True),
            StructField("Forename", StringType(), True),
            StructField("Surname", StringType(), True),
            StructField("DOB", StringType(), True)
        ])

        test_data = [
          [1, "Homer", "Simpson", "1983-05-12"],
          [2, "Marge", "Simpson", "1993-03-19"],
          [3, "Bart", "Simpson", "2012-04-01"],
          [4, "Lisa", "Simpson", "2014-05-09"]
        ]

        test_df = spark.createDataFrame(test_data, test_schema)

        expected_schema = StructType([
            StructField("ID", LongType(), True),
            StructField("Forename", StringType(), True),
            StructField("Surname", StringType(), True),
            StructField("DOB", StringType(), True),
            StructField("DoB_age_at_2022-11-03", IntegerType(), True),
        ])

        expected_data = [
          [1, "Homer", "Simpson", "1983-05-12", 39],
          [2, "Marge", "Simpson", "1993-03-19", 29],
          [3, "Bart", "Simpson", "2012-04-01", 10],
          [4, "Lisa", "Simpson", "2014-05-09", 8]
        ]

        intended_df = spark.createDataFrame(expected_data, expected_schema)      

        dates = ['2022-11-03']
        result_df = age_at(test_df,'DoB','yyyy-MM-dd',*dates)
        assert_df_equality(intended_df, result_df,
                           ignore_row_order=True, ignore_column_order=True)


##############################################################################


class TestReplace(object):
    def test_expected(self, spark):

        test_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "before": ["a", None, "c", ""],
                        "before1": ["a", "b", "c", "d"],
                        "after": [None, None, "f", ""],
                        "after1": [None, "b", "f", "d"],
                    }
                )
            )
        )

        intended_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "before": [None, None, "f", ""],
                        "before1": ["a", "b", "c", "d"],
                        "after": [None, None, "f", ""],
                        "after1": [None, "b", "f", "d"],
                    }
                )
            )
        )

        result_df = replace(
            test_df, subset="before", replace_dict={"a": None, "c": "f"}
        )
        assert_df_equality(intended_df, result_df,
                           ignore_row_order=True, ignore_column_order=True)

        intended_df2 = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "before": [None, None, "f", ""],
                        "before1": [None, "b", "f", "d"],
                        "after": [None, None, "f", ""],
                        "after1": [None, "b", "f", "d"],
                    }
                )
            )
        )

        result_df2 = replace(
            test_df, subset=["before", "before1"], replace_dict={"a": None, "c": "f"}
        )
        assert_df_equality(intended_df2, result_df2,
                           ignore_row_order=True, ignore_column_order=True)

    def test_expected_with_join(self, spark):

        test_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "before": ["a", None, "c", ""],
                        "before1": ["a", "b", "c", "d"],
                        "after": [None, None, "f", ""],
                        "after1": [None, "b", "f", "d"],
                    }
                )
            )
        )

        print("Test")
        test_df.show()
        intended_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "before": ["A", None, "f", ""],
                        "before1": ["a", "b", "c", "d"],
                        "after": [None, None, "f", ""],
                        "after1": [None, "b", "f", "d"],
                    }
                )
            )
        )

        print("Intended")
        intended_df.show()
        result_df = replace(
            test_df, subset="before", replace_dict={"a": "A", "c": "f"}, use_join=True
        )
        print("Result")
        result_df.show()
        assert_df_equality(intended_df, result_df,
                           ignore_row_order=True, ignore_column_order=True)

    def test_expected_with_regex(self, spark):

        test_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "before": ["alan", None, "betty", ""],
                        "before1": ["a", "b", "c", "d"],
                        "after": [None, None, "f", ""],
                        "after1": [None, "b", "f", "d"],
                    }
                )
            )
        )

        intended_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "before": ["A", None, "Y", ""],
                        "before1": ["a", "b", "c", "d"],
                        "after": [None, None, "f", ""],
                        "after1": [None, "b", "f", "d"],
                    }
                )
            )
        )

        result_df = replace(
            test_df,
            subset="before",
            replace_dict={"^a": "A", "y$": "Y"},
            use_regex=True
        )
        assert_df_equality(intended_df, result_df,
                           ignore_row_order=True, ignore_column_order=True)

    def test_value_error_on_join_and_regex(self, spark):
        test_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "before": ["alan", None, "betty", ""],
                        "before1": ["a", "b", "c", "d"],
                        "after": [None, None, "f", ""],
                        "after1": [None, "b", "f", "d"],
                    }
                )
            )
        )

        with pytest.raises(ValueError) as e:
            result_df = replace(
                test_df,
                subset="before",
                replace_dict={"^a": "A", "y$": "Y"},
                use_regex=True,
                use_join=True
            )

    def test_value_error_on_join_and_none(self, spark):
        test_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "before": ["alan", None, "betty", ""],
                        "before1": ["a", "b", "c", "d"],
                        "after": [None, None, "f", ""],
                        "after1": [None, "b", "f", "d"],
                    }
                )
            )
        )

        with pytest.raises(ValueError) as e:
            result_df = replace(
                test_df,
                subset="before",
                replace_dict={"^a": "A", None: "Y"},
                use_join=True
            )

##############################################################################


class TestCleanForename(object):
    def test_expected(self, spark):

        test_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "before": ["MISS Maddie", "MR GEORGE", "DR Paul", "NO NAME"],
                        "after": [" Maddie", " GEORGE", " Paul", ""],
                    }
                )
            )
        )

        intended_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "before": [" Maddie", " GEORGE", " Paul", ""],
                        "after": [" Maddie", " GEORGE", " Paul", ""],
                    }
                )
            )
        )

        result_df = clean_forename(test_df, "before")
        assert_df_equality(intended_df, result_df,
                           ignore_row_order=True, ignore_column_order=True)


##############################################################################


class TestCleanSurname(object):
    def test_expected(self, spark):

        test_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "before": ["O Leary", "VAN DER VAL", "SURNAME", "MC CREW"],
                        "after": ["OLeary", "VANDERVAL", "", "MCCREW"],
                    }
                )
            )
        )

        intended_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "before": ["OLeary", "VANDERVAL", "", "MCCREW"],
                        "after": ["OLeary", "VANDERVAL", "", "MCCREW"],
                    }
                )
            )
        )

        result_df = clean_surname(test_df, "before")
        assert_df_equality(intended_df, result_df,
                           ignore_row_order=True, ignore_column_order=True)


##############################################################################


class TestRegReplace(object):
    def test_expected(self, spark):

        test_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "col1": [None, "hello str", "king strt", "king road"],
                        "col2": [None, "bond street", "queen street", "queen avenue"],
                    }
                )
            )
        )

        intended_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "col1": [None, "bond street", "queen street", "queen avenue"],
                        "col2": [None, "bond street", "queen street", "queen avenue"],
                    }
                )
            )
        )

        result_df = reg_replace(
            test_df,
            replace_dict={
                "street": "\\bstr\\b|\\bstrt\\b",
                "avenue": "road",
                "bond": "hello",
                "queen": "king",
            },
        )
        assert_df_equality(intended_df, result_df,
                           ignore_row_order=True, ignore_column_order=True)
