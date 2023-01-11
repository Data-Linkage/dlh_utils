from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pandas as pd
from dlh_utils.standardisation import *
from dlh_utils.dataframes import *


def test_cast_type():
    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(
        (pd.DataFrame({
            "before": [None, '2', '3', '4', '5'],
            "after": [None, 2, 3, 4, 5]
        })))

    # check if it is string first
    assert (cast_type(df, subset='after', types='string')
            .select('after')
            .dtypes[0][1]) == 'string'

    # check if columns are the same after various conversions
    assert (cast_type(df, subset='before', types='int')
            .where(F.col("before") == F.col("after")).count() == 4
            )

##############################################################################


def test_standardise_white_space():
    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(
        (pd.DataFrame({
            "before": [None, 'hello  yes', 'hello yes', 'hello   yes', 'hello yes'],
            "after": [None, 'hello yes', 'hello yes', 'hello yes', 'hello yes'],

            "before2": [None, 'hello  yes', 'hello yes', 'hello   yes', 'hello yes'],
            "after2": [None, 'hello_yes', 'hello_yes', 'hello_yes', 'hello_yes']
        })))

    assert (standardise_white_space(df, subset='before', wsl='one').
            where(F.col("before") == F.col("after")).count() == 4)

    assert (standardise_white_space(df, subset='before2', fill='_').
            where(F.col("before2") == F.col("after2")).count() == 4)


##############################################################################

def test_remove_punct():
    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(
        (pd.DataFrame({
            "after": ['ONE', 'TWO', 'THREE', 'FOUR', 'FI^VE'],
            "before": [None, 'TW""O', "TH@REE", 'FO+UR', "FI@^VE"],
            "extra": [None, 'TWO', "TH@REE", 'FO+UR', "FI@^VE"]
        })))

    assert (remove_punct(df, keep='^')
            .where(F.col("after") == F.col("before")).count() == 4)

    assert (remove_punct(df, keep='^', subset=['after', 'before'])
            .where(F.col("after") == F.col("extra")).count() == 1)


##############################################################################

def test_trim():
    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(
        (pd.DataFrame({
            "before1": [None, '', ' th re e', '  four ', '  f iv  e '],
            "before2": [None, ' ', ' th re e', '  four ', '  f iv  e '],
            "numeric": [1, 2, 3, 4, 5],
            "after": [None, '', 'th re e', 'four', 'f iv  e']
        })))

    assert (trim(df)
            .where(((F.col("before1") == F.col("after"))
                    & (F.col("before2") == F.col("after")))
                   | ((F.col("before1").isNull() & F.col("after").isNull())
                      & ((F.col("before2").isNull() & F.col("after").isNull()))))
            .count()
            ) == 5

    assert df.dtypes == trim(df).dtypes

    assert (trim(df, subset=["before1", "numeric"])
            .where(((F.col("before1") == F.col("after"))
                    & (F.col("before2") != F.col("after")))
                   | ((F.col("before1").isNull() & F.col("after").isNull())
                      & ((F.col("before2").isNull() & F.col("after").isNull()))))
            .count()
            ) == 5

    assert df.dtypes == trim(df, subset=["before1", "numeric"]).dtypes

    assert (trim(df, subset="before1")
            .where(((F.col("before1") == F.col("after"))
                    & (F.col("before2") != F.col("after")))
                   | ((F.col("before1").isNull() & F.col("after").isNull())
                      & ((F.col("before2").isNull() & F.col("after").isNull()))))
            .count()
            ) == 5

    assert df.dtypes == trim(df, subset="numeric").dtypes


##############################################################################

def test_standardise_case():
    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(
        (pd.DataFrame({
            "upper": ['ONE', 'TWO', 'THREE'],
            "lower": ['one', 'two', 'three'],
            "title": ['One', 'Two', 'Three']
        })))

    assert (standardise_case(df, subset='lower', val='upper')
            .where(F.col('upper') == F.col('lower')).count() == 3)

    assert (standardise_case(df, subset='upper', val='lower')
            .where(F.col('upper') == F.col('lower')).count() == 3)

    assert (standardise_case(df, val='title')
            .where(F.col('upper') == F.col('title')).count() == 3)

    assert (standardise_case(df, val='title')
            .where(F.col('lower') == F.col('title')).count() == 3)

##############################################################################


def test_standardise_date():
    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(
        (pd.DataFrame({
            "before": [None, '14-05-1996', '15-04-1996'],
            "after": [None, '1996-05-14', '1996-04-15'],
            "slashed": [None, '14/05/1996', '15/04/1996'],
            "slashedReverse": [None, '1996/05/14', '1996/04/15']
        })))

    assert (standardise_date(df, col_name='before').
            where(F.col("before") == F.col("after")).count() == 2)

    assert (standardise_date(df, col_name='before', out_date_format='dd/mm/yyyy').
            where(F.col("before") == F.col("slashed")).count() == 2)

    assert (standardise_date(df, col_name='before', out_date_format='yyyy/mm/dd').
            where(F.col("before") == F.col("slashedReverse")).count() == 2)

    assert (standardise_date(df, col_name='slashed', in_date_format='dd/mm/yyyy',\ 
                             out_date_format='yyyy/mm/dd').
            where(F.col("slashed") == F.col("slashedReverse")).count() == 2)

    assert (standardise_date(df, col_name='slashedReverse', in_date_format='yyyy/mm/dd',\
                             out_date_format='dd-mm-yyyy').
            where(F.col("slashedReverse") == F.col("before")).count() == 2)


##############################################################################

def test_max_hyphen():

    spark = SparkSession.builder.getOrCreate()
    # max hyphen gets rid of any hyphens that does not match
    # or is under the limit
    df = spark.createDataFrame(
        pd.DataFrame({
            "before": ["james--brad", "tom----ridley", "chicken-wing", "agent-----john"],
            "after2": ["james--brad", "tomridley", "chicken-wing", "agentjohn"],
            "after4": ['james--brad', "tom----ridley", "chicken-wing", "agentjohn"]

        }))
    assert (max_hyphen(df, limit=2, subset=['before'])
            .where(F.col("before") == F.col("after2")).count() == 4)

    assert (max_hyphen(df, limit=4, subset=['before'])
            .where(F.col("before") == F.col("after4")).count() == 4)


##############################################################################

def test_max_white_space():

    spark = SparkSession.builder.getOrCreate()
    # max_white_space gets rid of any whitespace that does not match
    # or is under the limit
    df = spark.createDataFrame(
        pd.DataFrame({
            "before": ["james  brad", "tom    ridley", "chicken wing", "agent     john"],
            "after2": ["james  brad", "tomridley", "chicken wing", "agentjohn"],
            "after4": ['james  brad', "tom    ridley", "chicken wing", "agentjohn"]

        }))
    assert (max_white_space(df, limit=2, subset=['before'])
            .where(F.col("before") == F.col("after2")).count() == 4)

    assert (max_white_space(df, limit=4, subset=['before'])
            .where(F.col("before") == F.col("after4")).count() == 4)

##############################################################################


def test_align_forenames():

    spark = SparkSession.builder.getOrCreate()
    df1 = spark.createDataFrame(
        (pd.DataFrame({
            "identifier": [1, 2, 3, 4],
            "firstName": ["robert green", "andrew", 'carlos senior', "john wick"],
            "middleName": [None, "hog", None, ""]
        })))

    df2 = spark.createDataFrame(
        (pd.DataFrame({
            "identifier": [1, 2, 3, 4],
            "firstName": ["robert", "andrew", "carlos", "john"],
            "middleName": ["green", "hog", "senior", "wick"]
        })))

    assert (((align_forenames(df1, "firstName", "middleName", "identifier"))
             .intersect(df2)).count() == 4
            )

##############################################################################


def test_add_leading_zeros():

    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(
        (pd.DataFrame({
            "before1": ["1-2-12", "2-2-12", "3-2-12", "4-2-12", None],
            "after1": ["01-2-12", "02-2-12", "03-2-12", "04-2-12", None]
        })))

    # showing weird results
    #add_leading_zeros(df, cols=['before1'],n=2).show()
    #add_leading_zeros(df, cols=['before1'],n=4).show()

    #add_leading_zeros(df, cols=['before1'],n=7).show()
    # this works as the string is 6 chars and putting it to 7
    # makes it so that it knows that it needs to add only 1
    # zero to the start of the string

    assert (add_leading_zeros(df, cols=['before1'], n=7)
            .where(F.col("before1") == F.col("after1")).count() == 4)

##############################################################################


def test_group_single_characters():

    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(
        (pd.DataFrame({
            "before1": [None, '', '-t-h r e e', 'four', 'f i v e'],
            "before2": [None, '', '-t-h r e e', 'four', 'f i v e'],
            "after": [None, '', '-t-h ree', 'four', 'five']
        })))

    assert (group_single_characters(df, subset=None)
            .fillna('<<>>')
            .where(F.col('before1') == F.col('after'))
            .where(F.col('before2') == F.col('after'))
            ).count() == 5

    assert (group_single_characters(df, subset='before1')
            .fillna('<<>>')
            .where(F.col('before1') == F.col('after'))
            ).count() == 5

    assert (group_single_characters(df, subset='before1')
            .fillna('<<>>')
            .where(F.col('before2') == F.col('after'))
            ).count() == 3


##############################################################################

def test_clean_hyphens():

    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(
        (pd.DataFrame({
            "before1": [None, '', 'th- ree', '--fo - ur', 'fi -ve-'],
            "before2": [None, '', 'th- ree', 'fo - ur', 'fi -ve'],
            "after": [None, '', 'th-ree', 'fo-ur', 'fi-ve']
        })))

    assert (clean_hyphens(df, subset=None)
            .fillna('<<>>')
            .where(F.col('before1') == F.col('after'))
            .where(F.col('before2') == F.col('after'))
            ).count() == 5

    assert (clean_hyphens(df, subset='before1')
            .fillna('<<>>')
            .where(F.col('before1') == F.col('after'))
            ).count() == 5

    assert (clean_hyphens(df, subset='before1')
            .fillna('<<>>')
            .where(F.col('before2') == F.col('after'))
            ).count() == 2

##############################################################################


def test_standardise_null():

    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(
        (pd.DataFrame({
            "before1": [None, '', '  ', '-999', '####', 'KEEP'],
            "before2": [None, '', '  ', '-999', '####', 'KEEP'],
            "after": [None, None, None, None, None, 'KEEP']
        })))

    assert (standardise_null(df,
                             replace="^-[0-9]|^[#]+$|^$|^\s*$",
                             standard=None,
                             subset=None,
                             regex=True)
            .fillna('<<>>')
            .where(F.col('before1') == F.col('after'))
            .where(F.col('before2') == F.col('after'))
            ).count() == 6

    assert (standardise_null(df,
                             replace="^-[0-9]|^[#]+$|^$|^\s*$",
                             standard=None,
                             subset='before1',
                             regex=True)
            .fillna('<<>>')
            .where(F.col('before1') == F.col('after'))
            ).count() == 6

    assert (standardise_null(df,
                             replace="^-[0-9]|^[#]+$|^$|^\s*$",
                             standard=None,
                             subset='before1',
                             regex=True)
            .fillna('<<>>')
            .where(F.col('before2') == F.col('after'))
            ).count() == 2

    assert (standardise_null(df,
                             replace="-999",
                             standard=None,
                             subset='before1',
                             regex=False)
            .fillna('<<>>')
            .where(F.col('before1') == F.col('after'))
            ).count() == 3

##############################################################################


def test_cen_ethnic_group_5():

    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(
        (pd.DataFrame({
            "before": [None, '1500', '2002', '3600', '4002', '5002', '6420'],
            "after": [None, '1', '2', '3', '4', '4', '5']
        })))

    assert (cen_ethnic_group_5(df, "before")
            .fillna('<<>>')
            .where(F.col('before') == F.col('after'))
            ).count() == 7

##############################################################################

def test_fill_nulls():
    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(
        (pd.DataFrame({
            "before": ['abcd', None, 'fg', ''],
            "numeric": [1, 2, None, 3],
            "after": ['abcd', None, 'fg', ''],
            "afternumeric": [1, 2, 0, 3]
        })))

    assert (fill_nulls(df, 0)
            .where(((F.col("before") == F.col("after"))
                    & (F.col("numeric") == F.col("afternumeric")))
                   | ((F.col("before").isNull() & F.col("after").isNull())
                      & (F.col("numeric").isNull() & F.col("afternumeric").isNull())))
            .count()
            ) == 4

##############################################################################


def test_replace():
    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(
        (pd.DataFrame({
            "before": ['a', None, 'c', ''],
            "before1": ['a', 'b', 'c', 'd'],
            "after": [None, None, 'f', ''],
            "after1": [None, 'b', 'f', 'd']
        })))

    assert (replace(df, cols="before", replace_dict={'a': None,
                                                     'c': 'f'})
            .where((F.col("before") == F.col("after"))
                   | (F.col("before").isNull() & F.col("after").isNull()))
            .count()
            ) == 4

    assert (replace(df, cols=["before", "before1"], replace_dict={'a': None,
                                                                  'c': 'f'})
            .fillna("^&*&")
            .where(((F.col("before") == F.col("after"))
                    & (F.col("before1") == F.col("after1"))))
            .count()
            ) == 4

##############################################################################


def test_clean_forename():
    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(
        (pd.DataFrame({
            "before": ['MISS Maddie', 'MR GEORGE', 'DR Paul', 'NO NAME'],
            "after": [' Maddie', ' GEORGE', ' Paul', '']
        })))

    assert (clean_forename(df, "before")
            .where(F.col("before") == F.col("after"))
            .count()
            ) == 4

##############################################################################


def test_clean_surname():
    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(
        (pd.DataFrame({
            "before": ['O Leary', 'VAN DER VAL', 'SURNAME', 'MC CREW'],
            "after": ['OLeary', 'VANDERVAL', '', 'MCCREW']
        })))

    assert (clean_surname(df, "before")
            .where(F.col("before") == F.col("after"))
            .count()
            ) == 4

##############################################################################


def test_reg_replace():
    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(
        (pd.DataFrame({
            "col1": [None, "hello str", 'king strt', 'king road'],
            "col2": [None, "bond street", "queen street", "queen avenue"]
        })))

    assert (reg_replace(df, dic={'street': '\\bstr\\b|\\bstrt\\b',
                                'avenue': 'road',
                                "bond": "hello",
                                "queen": "king"})
            .where(F.col("col1") == F.col("col2"))
            .count() == 3)

##############################################################################


def test_cast_geography_null():

    spark = SparkSession.builder.getOrCreate()

    df = spark.createDataFrame(
        (pd.DataFrame({
            "postcode": ['zz99 lmn', 'ZZ99ABC', 'CF249ZZ', None, ""],
            "uprn": ['1', '2', '3', '4', '5'],
            "city": ['a', 'b', 'c', 'd', 'e']
        })))

    target_col = 'postcode'
    geo_cols = ['uprn', 'city']

    assert (cast_geography_null(df, target_col, geo_cols, regex="^(?i)zz99")
            .where(F.col('postcode') == 'zz99 lmn')
            ).count() == 0

    assert (cast_geography_null(df, target_col, geo_cols, regex="^(?i)zz99")
            .where(F.col('postcode') == 'ZZ99ABC')
            ).count() == 0

    assert (cast_geography_null(df, target_col, geo_cols, regex="^(?i)zz99")
            .where(F.col('postcode') == 'CF249ZZ')
            ).count() == 1

    assert (cast_geography_null(df, target_col, geo_cols, regex="^(?i)zz99")
            .where(F.col('uprn').isin(['1', '2']))
            ).count() == 0

    assert (cast_geography_null(df, target_col, geo_cols, regex="^(?i)zz99")
            .where(F.col('city').isin(['a', 'b']))
            ).count() == 0

    assert (cast_geography_null(df, target_col, geo_cols, regex="^(?i)zz99")
            .where(F.col('postcode').isNull())
            ).count() == 3

    target_col = 'uprn'
    geo_cols = ['postcode', 'city']

    assert (cast_geography_null(df, target_col, geo_cols, regex="^3$")
            .where(F.col('postcode') == 'CF249ZZ')
            ).count() == 0
