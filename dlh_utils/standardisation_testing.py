import pyspark
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pandas as pd
from dlh_utils.standardisation import *
from dlh_utils.dataframes import *
import chispa
from chispa import assert_df_equality
import pytest
from pyspark.sql.types import *


pytestmark = pytest.mark.usefixtures("spark")


@pytest.fixture(scope="session")
def spark(request):
    """ fixture for creating a spark context
    Args:
        request: pytest.FixtureRequest object
    """
    spark = (SparkSession.builder.appName("dataframe_testing")
             .config('spark.executor.memory', '5g')
             .config('spark.yarn.excecutor.memoryOverhead', '2g')
             .getOrCreate())
    request.addfinalizer(lambda: spark.stop())
    return spark


class TestCastType(object):
  
    def test_expected(self, spark):
      
        test_df = spark.createDataFrame(
            (pd.DataFrame({
                "before": [None, '2', '3', '4', '5'],
                "after": [None, 2, 3, 4, 5]
            })))

        intended_schema = StructType([
            StructField("after", StringType(), True),
            StructField("before", StringType(), True)
        ])

        intended_data = [[float('NaN'), None],
                         [2.0, 2],
                         [3.0, 3],
                         [4.0, 4],
                         [5.0, 5]]
        intended_df = spark.createDataFrame(intended_data, intended_schema)

        # check if it is string first
        result_df = cast_type(test_df, subset='after', types='string')
        assert_df_equality(intended_df, result_df, allow_nan_equality=True)

        intended_schema = StructType([
            StructField("after", DoubleType(), True),
            StructField("before", IntegerType(), True)
        ])

        intended_data = [[float('NaN'), None],
                         [2.0, 2],
                         [3.0, 3],
                         [4.0, 4],
                         [5.0, 5]]
        intended_df2 = spark.createDataFrame(intended_data, intended_schema)
        # check if columns are the same after various conversions
        result_df2 = cast_type(test_df, subset='before', types='int')
        assert_df_equality(intended_df2, result_df2, allow_nan_equality=True)
##############################################################################


class TestStandardiseWhiteSpace(object):
  
    def test_expected(self, spark):

        test_df = spark.createDataFrame(
            (pd.DataFrame({
                "before": [None, 'hello  yes', 'hello yes', 'hello   yes', 'hello yes'],
                "after": [None, 'hello yes', 'hello yes', 'hello yes', 'hello yes'],

                "before2": [None, 'hello  yes', 'hello yes', 'hello   yes', 'hello yes'],
                "after2": [None, 'hello_yes', 'hello_yes', 'hello_yes', 'hello_yes']
            })))
        intended_df = spark.createDataFrame(
            (pd.DataFrame({
                "before": [None, 'hello yes', 'hello yes', 'hello yes', 'hello yes'],
                "after": [None, 'hello yes', 'hello yes', 'hello yes', 'hello yes'],

                "before2": [None, 'hello  yes', 'hello yes', 'hello   yes', 'hello yes'],
                "after2": [None, 'hello_yes', 'hello_yes', 'hello_yes', 'hello_yes']
            })))
        result_df = standardise_white_space(
            test_df, subset='before', wsl='one')
        assert_df_equality(intended_df, result_df)

        intended_df2 = spark.createDataFrame(
            (pd.DataFrame({
                "before": [None, 'hello  yes', 'hello yes', 'hello   yes', 'hello yes'],
                "after": [None, 'hello yes', 'hello yes', 'hello yes', 'hello yes'],

                "before2": [None, 'hello_yes', 'hello_yes', 'hello_yes', 'hello_yes'],
                "after2": [None, 'hello_yes', 'hello_yes', 'hello_yes', 'hello_yes']
            })))
        result_df2 = standardise_white_space(
            test_df, subset='before2', fill='_')
        assert_df_equality(intended_df2, result_df2)


##############################################################################

class TestRemovePunct(object):
  
    def test_expected(self, spark):

        test_df = spark.createDataFrame(
            (pd.DataFrame({
                "after": ['ONE', 'TWO', 'THREE', 'FOUR', 'FI^VE'],
                "before": [None, 'TW""O', "TH@REE", 'FO+UR', "FI@^VE"],
                "extra": [None, 'TWO', "TH@REE", 'FO+UR', "FI@^VE"]
            })))

        intended_df = spark.createDataFrame(
            (pd.DataFrame({
                "after": ['ONE', 'TWO', 'THREE', 'FOUR', 'FI^VE'],
                "before": [None, 'TWO', "THREE", 'FOUR', "FI^VE"],
                "extra": [None, 'TWO', "TH@REE", 'FO+UR', "FI@^VE"]
            })))

        result_df = remove_punct(test_df, keep='^', subset=['after', 'before'])

        assert_df_equality(intended_df, result_df)

##############################################################################


class TestTrim(object):
  
    def test_expected(self, spark):

        test_df = spark.createDataFrame(
            (pd.DataFrame({
                "before1": [None, '', ' th re e', '  four ', '  f iv  e '],
                "before2": [None, ' ', ' th re e', '  four ', '  f iv  e '],
                "numeric": [1, 2, 3, 4, 5],
                "after": [None, '', 'th re e', 'four', 'f iv  e']
            })))

        intended_df = spark.createDataFrame(
            (pd.DataFrame({
                "before1": [None, '', 'th re e', 'four', 'f iv  e'],
                "before2": [None, ' ', ' th re e', '  four ', '  f iv  e '],
                "numeric": [1, 2, 3, 4, 5],
                "after": [None, '', 'th re e', 'four', 'f iv  e']
            })))

        result_df = trim(test_df, subset=["before1", "numeric", "after"])

        assert_df_equality(intended_df, result_df)


##############################################################################

class TestStandardiseCase(object):
  
    def test_expected(self, spark):

        test_df = spark.createDataFrame(
            (pd.DataFrame({
                "upper": ['ONE', 'TWO', 'THREE'],
                "lower": ['one', 'two', 'three'],
                "title": ['One', 'Two', 'Three']
            })))

        intended_df = spark.createDataFrame(
            (pd.DataFrame({
                "upper": ['ONE', 'TWO', 'THREE'],
                "lower": ['ONE', 'TWO', 'THREE'],
                "title": ['One', 'Two', 'Three']
            })))

        result_df = standardise_case(test_df, subset='lower', val='upper')
        assert_df_equality(intended_df, result_df)

        intended_df2 = spark.createDataFrame(
            (pd.DataFrame({
                "upper": ['one', 'two', 'three'],
                "lower": ['one', 'two', 'three'],
                "title": ['One', 'Two', 'Three']
            })))

        result_df2 = standardise_case(test_df, subset='upper', val='lower')
        assert_df_equality(intended_df2, result_df2)

##############################################################################


class TestStandardiseDate(object):
  
    def test_expected(self, spark):

        test_df = spark.createDataFrame(
            (pd.DataFrame({
                "before": [None, '14-05-1996', '15-04-1996'],
                "after": [None, '1996-05-14', '1996-04-15'],
                "slashed": [None, '14/05/1996', '15/04/1996'],
                "slashedReverse": [None, '1996/05/14', '1996/04/15']
            })))

        intended_df = spark.createDataFrame(
            (pd.DataFrame({
                "before": [None, '1996-05-14', '1996-04-15'],
                "after": [None, '1996-05-14', '1996-04-15'],
                "slashed": [None, '14/05/1996', '15/04/1996'],
                "slashedReverse": [None, '1996/05/14', '1996/04/15']
            })))

        result_df = standardise_date(test_df, col_name='before')
        assert_df_equality(intended_df, result_df)

        intended_df2 = spark.createDataFrame(
            (pd.DataFrame({
                "before": [None, '14/05/1996', '15/04/1996'],
                "after": [None, '1996-05-14', '1996-04-15'],
                "slashed": [None, '14/05/1996', '15/04/1996'],
                "slashedReverse": [None, '1996/05/14', '1996/04/15']
            })))

        result_df2 = standardise_date(
            test_df,
            col_name='before',
            out_date_format='dd/MM/yyyy')
        assert_df_equality(intended_df2, result_df2)

        intended_df3 = spark.createDataFrame(
            (pd.DataFrame({
                "before": [None, '14-05-1996', '15-04-1996'],
                "after": [None, '1996-05-14', '1996-04-15'],
                "slashed": [None, '14/05/1996', '15/04/1996'],
                "slashedReverse": [None, '14-05-1996', '15-04-1996']
            })))

        result_df3 = standardise_date(
            test_df,
            col_name='slashedReverse',
            in_date_format='yyyy/mm/dd',
            out_date_format='dd-mm-yyyy')

        assert_df_equality(intended_df3, result_df3)

##############################################################################


class TestMaxHyphen(object):
  
    def test_expected(self, spark):
      
        # max hyphen gets rid of any hyphens that does not match
        # or is under the limit
        test_df = spark.createDataFrame(
            pd.DataFrame({
                "before": ["james--brad", "tom----ridley", "chicken-wing", "agent-----john"],
                "after2": ["james--brad", "tomridley", "chicken-wing", "agentjohn"],
                "after4": ['james--brad', "tom----ridley", "chicken-wing", "agentjohn"]

            }))

        intended_df = spark.createDataFrame(
            pd.DataFrame({
                "before": ["james--brad", "tom--ridley", "chicken-wing", "agent--john"],
                "after2": ["james--brad", "tomridley", "chicken-wing", "agentjohn"],
                "after4": ['james--brad', "tom----ridley", "chicken-wing", "agentjohn"]

            }))
        result_df = max_hyphen(test_df, limit=2, subset=['before'])
        assert_df_equality(intended_df, result_df)

        intended_df2 = spark.createDataFrame(
            pd.DataFrame({
                "before": ["james--brad", "tom----ridley", "chicken-wing", "agent----john"],
                "after2": ["james--brad", "tomridley", "chicken-wing", "agentjohn"],
                "after4": ['james--brad', "tom----ridley", "chicken-wing", "agentjohn"]

            }))
        result_df2 = max_hyphen(test_df, limit=4, subset=['before'])
        assert_df_equality(intended_df2, result_df2)

##############################################################################


class TestMaxWhiteSpace(object):
  
    def test_expected(self, spark):

        # max_white_space gets rid of any whitespace that does not match
        # or is under the limit
        test_df = spark.createDataFrame(
            pd.DataFrame({
                "before": ["james  brad", "tom    ridley", "chicken wing", "agent     john"],
                "after2": ["james  brad", "tomridley", "chicken wing", "agentjohn"],
                "after4": ['james  brad', "tom    ridley", "chicken wing", "agentjohn"]

            }))

        intended_df = spark.createDataFrame(
            pd.DataFrame({
                "before": ["james  brad", "tomridley", "chicken wing", "agentjohn"],
                "after2": ["james  brad", "tomridley", "chicken wing", "agentjohn"],
                "after4": ['james  brad', "tom    ridley", "chicken wing", "agentjohn"]

            }))

        result_df = max_white_space(test_df, limit=2, subset=['before'])
        assert_df_equality(intended_df, result_df)

        intended_df2 = spark.createDataFrame(
            pd.DataFrame({
                "before": ["james  brad", "tom    ridley", "chicken wing", "agentjohn"],
                "after2": ["james  brad", "tomridley", "chicken wing", "agentjohn"],
                "after4": ['james  brad', "tom    ridley", "chicken wing", "agentjohn"]

            }))

        result_df2 = max_white_space(test_df, limit=4, subset=['before'])
        assert_df_equality(intended_df2, result_df2)

##############################################################################


class TestAlignForenames(object):
  
    def test_expected(self, spark):
      
        test_df = spark.createDataFrame(
            (pd.DataFrame({
                "identifier": [1, 2, 3, 4],
                "firstName": ["robert green", "andrew", 'carlos senior', "john wick"],
                "middleName": [None, "hog", None, ""]
            })))

        intended_df = spark.createDataFrame(
            (pd.DataFrame({
                "identifier": [1, 3, 2, 4],
                "firstName": ["robert", "carlos", "andrew", "john"],
                "middleName": ["green", "senior", "hog", "wick"]
            })))

        result_df = align_forenames(
            test_df, "firstName", "middleName", "identifier")
        assert_df_equality(intended_df, result_df)
##############################################################################


class TestAddLeadingZeros(object):

    def test_expected(self, spark):
        test_df = spark.createDataFrame(
            (pd.DataFrame({
                "before1": ["1-2-12", "2-2-12", "3-2-12", "4-2-12", None],
                "after1": ["01-2-12", "02-2-12", "03-2-12", "04-2-12", None]
            })))

        intended_df = spark.createDataFrame(
            (pd.DataFrame({
                "before1": ["01-2-12", "02-2-12", "03-2-12", "04-2-12", None],
                "after1": ["01-2-12", "02-2-12", "03-2-12", "04-2-12", None]
            })))

        result_df = add_leading_zeros(test_df, subset=['before1'], n=7)
        assert_df_equality(intended_df, result_df)
##############################################################################


class TestGroupSingleCharacters(object):
  
    def test_expected(self, spark):

        test_df = spark.createDataFrame(
            (pd.DataFrame({
                "before1": [None, '', '-t-h r e e', 'four', 'f i v e'],
                "before2": [None, '', '-t-h r e e', 'four', 'f i v e'],
                "after": [None, '', '-t-h ree', 'four', 'five']
            })))

        intended_df = spark.createDataFrame(
            (pd.DataFrame({
                "before1": [None, '', '-t-h ree', 'four', 'five'],
                "before2": [None, '', '-t-h r e e', 'four', 'f i v e'],
                "after": [None, '', '-t-h ree', 'four', 'five']
            })))

        result_df = group_single_characters(test_df, subset='before1')
        assert_df_equality(intended_df, result_df)


##############################################################################

class TestCleanHyphens(object):
  
    def test_expected(self, spark):
      
        test_df = spark.createDataFrame(
            (pd.DataFrame({
                "before1": [None, '', 'th- ree', '--fo - ur', 'fi -ve-'],
                "before2": [None, '', 'th- ree', 'fo - ur', 'fi -ve'],
                "after": [None, '', 'th-ree', 'fo-ur', 'fi-ve']
            })))

        intended_df = spark.createDataFrame(
            (pd.DataFrame({
                "before1": [None, '', 'th-ree', 'fo-ur', 'fi-ve'],
                "before2": [None, '', 'th- ree', 'fo - ur', 'fi -ve'],
                "after": [None, '', 'th-ree', 'fo-ur', 'fi-ve']
            })))

        result_df = clean_hyphens(test_df, subset='before1')
        assert_df_equality(intended_df, result_df)


#############################################################################


class TestStandardiseNull(object):
  
    def test_expected(self, spark):
      
        test_df = spark.createDataFrame(
            (pd.DataFrame({
                "before1": [None, '', '  ', '-999', '####', 'KEEP'],
                "before2": [None, '', '  ', '-999', '####', 'KEEP'],
                "after": [None, None, None, None, None, 'KEEP']
            })))

        intended_df = spark.createDataFrame(
            (pd.DataFrame({
                          "before1": [None, None, None, None, None, 'KEEP'],
                          "before2": [None, '', '  ', '-999', '####', 'KEEP'],
                          "after": [None, None, None, None, None, 'KEEP']
                          })))

        result_df = standardise_null(test_df,
                                     replace="^-[0-9]|^[#]+$|^$|^\\s*$",
                                     subset='before1',
                                     regex=True)
        assert_df_equality(intended_df, result_df)

        intended_df2 = spark.createDataFrame(
            (pd.DataFrame({
                "before1": [None, '', '  ', None, '####', 'KEEP'],
                "before2": [None, '', '  ', '-999', '####', 'KEEP'],
                "after": [None, None, None, None, None, 'KEEP']
            })))

        result_df2 = standardise_null(test_df,
                                      replace="-999",
                                      subset='before1',
                                      regex=False)
        assert_df_equality(intended_df2, result_df2)


##############################################################################


class TestFillNulls(object):
  
    def test_expected(self, spark):
      
        test_df = spark.createDataFrame(
            (pd.DataFrame({
                "before": ['abcd', None, 'fg', ''],
                "numeric": [1, 2, None, 3],
                "after": ['abcd', None, 'fg', ''],
                "afternumeric": [1, 2, 0, 3]
            })))

        intended_df = spark.createDataFrame(
            (pd.DataFrame({
                "before": ['abcd', '0', 'fg', ''],
                "numeric": [1.0, 2.0, 0.0, 3.0],
                "after": ['abcd', '0', 'fg', ''],
                "afternumeric": [1, 2, 0, 3]
            })))

        result_df = fill_nulls(test_df, 0)
        assert_df_equality(intended_df, result_df)

##############################################################################


class TestReplace(object):
  
    def test_expected(self, spark):

        test_df = spark.createDataFrame(
            (pd.DataFrame({
                "before": ['a', None, 'c', ''],
                "before1": ['a', 'b', 'c', 'd'],
                "after": [None, None, 'f', ''],
                "after1": [None, 'b', 'f', 'd']
            })))

        intended_df = spark.createDataFrame(
            (pd.DataFrame({
                "before": [None, None, 'f', ''],
                "before1": ['a', 'b', 'c', 'd'],
                "after": [None, None, 'f', ''],
                "after1": [None, 'b', 'f', 'd']
            })))

        result_df = replace(test_df, subset="before", replace_dict={'a': None,
                                                                    'c': 'f'})
        assert_df_equality(intended_df, result_df)

        intended_df2 = spark.createDataFrame(
            (pd.DataFrame({
                "before": [None, None, 'f', ''],
                "before1": [None, 'b', 'f', 'd'],
                "after": [None, None, 'f', ''],
                "after1": [None, 'b', 'f', 'd']
            })))

        result_df2 = replace(
            test_df,
            subset=[
                "before",
                "before1"],
            replace_dict={
                'a': None,
                'c': 'f'})
        assert_df_equality(intended_df2, result_df2)


##############################################################################


class TestCleanForename(object):
  
    def test_expected(self, spark):
      
        test_df = spark.createDataFrame(
            (pd.DataFrame({
                "before": ['MISS Maddie', 'MR GEORGE', 'DR Paul', 'NO NAME'],
                "after": [' Maddie', ' GEORGE', ' Paul', '']
            })))

        intended_df = spark.createDataFrame(
            (pd.DataFrame({
                "before": [' Maddie', ' GEORGE', ' Paul', ''],
                "after": [' Maddie', ' GEORGE', ' Paul', '']
            })))

        result_df = clean_forename(test_df, "before")
        assert_df_equality(intended_df, result_df)


##############################################################################


class TestCleanSurname(object):
  
    def test_expected(self, spark):
      
        test_df = spark.createDataFrame(
            (pd.DataFrame({
                "before": ['O Leary', 'VAN DER VAL', 'SURNAME', 'MC CREW'],
                "after": ['OLeary', 'VANDERVAL', '', 'MCCREW']
            })))

        intended_df = spark.createDataFrame(
            (pd.DataFrame({
                "before": ['OLeary', 'VANDERVAL', '', 'MCCREW'],
                "after": ['OLeary', 'VANDERVAL', '', 'MCCREW']
            })))

        result_df = clean_surname(test_df, "before")
        assert_df_equality(intended_df, result_df)

##############################################################################


class TestRegReplace(object):
  
    def test_expected(self, spark):
      
        test_df = spark.createDataFrame(
            (pd.DataFrame({
                "col1": [None, "hello str", 'king strt', 'king road'],
                "col2": [None, "bond street", "queen street", "queen avenue"]
            })))

        intended_df = spark.createDataFrame(
            (pd.DataFrame({
                "col1": [None, "bond street", 'queen street', 'queen avenue'],
                "col2": [None, "bond street", 'queen street', "queen avenue"]
            })))

        result_df = reg_replace(test_df, dic={'street': '\\bstr\\b|\\bstrt\\b',
                                              'avenue': 'road',
                                              "bond": "hello",
                                              "queen": "king"})
        assert_df_equality(intended_df, result_df)


##############################################################################


class TestCastGeographyNull(object):

    def test_expected(self, spark):

        test_df = spark.createDataFrame(
            (pd.DataFrame({
                "postcode": ['zz99 lmn', 'ZZ99ABC', 'CF249ZZ', None, ""],
                "uprn": ['1', '2', '3', '4', '5'],
                "city": ['a', 'b', 'c', 'd', 'e']
            })))

        target_col = 'postcode'
        geo_cols = ['uprn', 'city']

        intended_df = spark.createDataFrame(
            (pd.DataFrame({
                "postcode": [None, None, 'CF249ZZ', None, ""],
                "uprn": [None, None, '3', '4', '5'],
                "city": [None, None, 'c', 'd', 'e']
            })))

        result_df = cast_geography_null(
            test_df, target_col, geo_cols=geo_cols, regex="^(?i)zz99")
        assert_df_equality(intended_df, result_df)

        target_col2 = 'uprn'
        geo_cols2 = ['postcode', 'city']

        intended_df2 = spark.createDataFrame(
            (pd.DataFrame({
                "postcode": ['zz99 lmn', 'ZZ99ABC', None, None, ""],
                "uprn": ['1', '2', None, '4', '5'],
                "city": ['a', 'b', None, 'd', 'e']
            })))

        result_df2 = cast_geography_null(
            test_df, target_col2, geo_cols=geo_cols2, regex="^3$")
        assert_df_equality(intended_df2, result_df2)
