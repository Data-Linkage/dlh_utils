import pyspark
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pandas as pd
from dlh_utils.standardisation import *
from dlh_utils.dataframes import *


#############################################################################

def test_explode():
    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(
        (pd.DataFrame({
            "before1": ['a_b_c'],
            "check": ['iagsigajs']
        })))

    assert explode(df, "before1", '_').count() == 3

    assert list(explode(df, "before1", '_')
                .toPandas()
                .sort_values('before1')['before1']
                ) == ['a', 'b', 'c']

    assert (explode(df, "before1", '_')
            .select('check')
            .dropDuplicates()
            .count()
            ) == 1


#############################################################################

def test_concat():

    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(
        (pd.DataFrame({
            "firstname": [None, 'Claire', 'Josh', 'Bob'],
            "middlename": ['Maria', None, '', 'Greg'],
            "lastname": ['Jones', None, 'Smith', 'Evans'],
            "numeric": [1, 2, None, 4],
            "after": ['Maria_Jones', 'Claire', 'Josh_Smith', 'Bob_Greg_Evans']
        })))

    assert (concat(df, 'fullname', sep="_", cols=["firstname",
                                                  "middlename", "lastname"])
            .where((F.col("fullname") == F.col("after")))
            .count()
            ) == 4

##############################################################################


def test_dropColumns():

    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(
        (pd.DataFrame({
            "col1": ['ONE', 'TWO', 'THREE'],
            "col2": ['one', 'two', 'three'],
            "extra": ['One', 'Two', 'Three']
        })))

    assert (len(dropColumns(df, subset='col1').columns) == 2)
    assert (len(dropColumns(df, subset=['col1', 'col2']).columns) == 1)
    assert (len(dropColumns(df, startswith='col').columns) == 1)
    assert (len(dropColumns(df, startswith='ex').columns) == 2)
    assert (len(dropColumns(df, endswith='1').columns) == 2)
    assert (len(dropColumns(df, endswith='tra').columns) == 2)


##############################################################################

def test_select():

    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(
        (pd.DataFrame({
            "identifier": [1, 2, 3, 4],
            "firstName": ["robert", "andrew", "carlos", "john"],
            "firstLetter": ["r", "a", "c", "j"],
            "first": ['x', '2', '3', '4'],
            "numbers": [1, 2, 3, 4],
        })))
    assert (len(select(df, columns=['identifier', 'firstName']).columns) == 2)
    assert (len(select(df, startswith='first').columns) == 3)
    assert (len(select(df, endswith='ers').columns) == 1)
    assert (len(select(df, contains='ame').columns) == 1)


################################################################################

def test_coalesced():

    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(
        (pd.DataFrame({
            "lower": ['one', None, 'one', 'four', None],
            "value": [1, 2, 3, 4, 5],
            "extra": [None, None, None, 'FO+ UR', None],
            "lowerNulls": ['one', 'two', None, 'four', None],
            "upperNulls": ["ONE", 'TWO', None, 'FOU  R', None]
        })))

    assert (coalesced(df)
            .where(F.col("coalescedCol").isNull())
            .count() == 0)

#################################################################


def test_cutOff():

    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(
        (pd.DataFrame({
            "strings": [None, '2', '3', '4', '5'],
            "ints": [None, 2, 3, 4, 5]
        })))

    # cutOff does not remove null values when the val is an Int type
    assert (cutOff(df, threshold_column='ints', val=3, mode='>=').count() == 4)

    # cutOff removes null values when the val is a string type
    assert (cutOff(df, threshold_column='strings',
            val='3', mode='>=').count() == 3)

    assert (cutOff(df, threshold_column='strings',
            val='3', mode='>').count() == 2)
    assert (cutOff(df, threshold_column='strings',
            val='2', mode='<=').count() == 1)
    assert (cutOff(df, threshold_column='strings',
            val='4', mode='<').count() == 2)

    df = spark.createDataFrame(
        (pd.DataFrame({
            "col1": [None, '15-05-1996', '16-04-1996', '17-06-1996', '18-05-1997']
        })))
    df2 = df.withColumn("col1", F.to_date("col1", 'dd-MM-yyyy'))

    assert (cutOff(df2, 'col1', '1997-01-15', '>=').count() == 1)
    assert (cutOff(df2, 'col1', '1997-01-15', '<=').count() == 3)
    assert (cutOff(df2, 'col1', '1996-05-15', '<=').count() == 2)

####################################################################


def test_literalColumn():

    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(
        (pd.DataFrame({
            "col1": ['one', None, 'one', 'four', None],
            "col2": [1, 2, 3, 4, 5]
        })))

    assert (literalColumn(df, "newStr", "yes")
            .where(F.col("newStr") == "yes").count() == 5)

    assert (literalColumn(df, "newInt", 1)
            .where(F.col("newInt") == 1).count() == 5)


####################################################################

def test_dropNulls():

    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(
        (pd.DataFrame({
            "lower": [None, None, 'one', 'four', 'five'],
            "after": [None, None, 'one', 'four', 'three']
        })))

    assert (dropNulls(df)
            .count() == 3)

    assert (dropNulls(df, val='five').count() == 4)

    assert (dropNulls(df, subset='lower', val='five').count() == 4)


def test_union_all():
    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(
        (pd.DataFrame({
            "col1": [None, None, 'one', 'four', 'five'],
            "col2": [None, None, 'one', 'four', 'three']
        })))

    df2 = spark.createDataFrame(
        (pd.DataFrame({
            "col1": [None, 'okay', 'dfs', 'few', 'dfs'],
            "col2": [None, None, 'fdsa', 'rew', 'trt']
        })))
    df3 = spark.createDataFrame(
        (pd.DataFrame({
            "col3": [None, 'okay', 'dfs', 'few', 'dfs']
        })))
    assert (union_all(df, df2).count() == 10)

    assert (union_all(df, df2, df3, fill='xd').
            where(F.col('col1') == 'xd').count() == 5)

    assert (union_all(df, df2, df3, fill='xd').
            where(F.col('col2') == 'xd').count() == 5)

#########################################################################


def test_rename_columns():

    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(
        (pd.DataFrame({
            "col1": [None, None, 'one', 'four', 'five'],
            "col2": [None, None, 'one', 'four', 'five']
        })))

    assert (rename_columns(df, rename_dict={"col1": "first", "col2": "second"})
            .where(F.col("first") == F.col("second")).count() == 3)

#########################################################################


def test_rename_columns2():
    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(
        (pd.DataFrame({
            "abefore": [['a', 'b', 'c'], None, ['b', 'c', 'd']],
            "bbefore": ['a', None, 'b'],
            "cbefore": ['c', None, 'd']
        })))

    assert (sorted(rename_columns(df, rename_dict={"abefore": 'aafter',
                                                   'bbefore': 'bafter',
                                                   'cbefore': 'cafter'})
                   .columns)) == ['aafter', 'bafter', 'cafter']

#########################################################################


def test_prefix_columns():

    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(
        (pd.DataFrame({
            "col1": [None, None, 'one', 'four', 'five'],
            "col2": [None, None, 'one', 'four', 'five']
        })))

    assert (prefix_columns(df, prefix='mr').
            where(F.col("mrcol1") == F.col("mrcol2")).count() == 3)

    assert (prefix_columns(df, prefix='mr', exclude='col1').
            where(F.col("col1") == F.col("mrcol2")).count() == 3)

   ###########################################################################


def test_suffix_columns():

    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(
        (pd.DataFrame({
            "col1": [None, None, 'one', 'four', 'five'],
            "col2": [None, None, 'one', 'four', 'five']
        })))

    assert (suffix_columns(df, suffix='mr').
            where(F.col("col1mr") == F.col("col2mr")).count() == 3)

    assert (suffix_columns(df, suffix='mr', exclude='col1').
            where(F.col("col1") == F.col("col2mr")).count() == 3)

#######################################################################


def test_window():
    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(
        (pd.DataFrame({
            "col1": ['a', 'b', 'c', 'c', 'd', 'e', 'd'],
            "col2": [1, 1, 2, 2, 1, 1, 1]
        })))

    assert (window(df, window=['col1', 'col2'], target='col2', mode='count', alias='new')
            .where((F.col("col1") == 'd') & (F.col("col2") == 1) & (F.col("new") == 2)).count() == 2)

    assert (window(df, window=['col1', 'col2'], target='col2', mode='count', alias='new')
            .where((F.col("col1") == 'c') & (F.col("col2") == 2) & (F.col("new") == 2)).count() == 2)

    df = spark.createDataFrame(
        (pd.DataFrame({
            "col1": ['a', 'b', 'c', 'c', 'd', 'e', 'd'],
            "col2": [1, 1, 1, 2, 1, 1, 2]
        })))

    assert (window(df, window=['col1'], target='col2', mode='min', alias='new')
            .where((F.col("col1") == 'c') & (F.col("new") == 1)).count() == 2)

    assert (window(df, window=['col1'], target='col2', mode='max', alias='new')
            .where((F.col("col1") == 'c') & (F.col("new") == 2)).count() == 2)
    df = spark.createDataFrame(
        (pd.DataFrame({
            "col1": ['a', 'b', 'c', 'c', 'd', 'e', 'd', 'c', 'c', 'c', 'd', 'd'],
            "col2": [1, 1, 1, 2, 1, 1, 2, 5, 6, 7, 11, 12]
        })))
    assert (window(df, window=['col1'], target='col2', mode='countDistinct', alias='new')
            .where((F.col("col1") == 'c') & (F.col("new") == 5)).count() == 5)

    assert (window(df, window=['col1'], target='col2', mode='countDistinct', alias='new')
            .where((F.col("col1") == 'd') & (F.col("new") == 4)).count() == 4)

#############################################################


def test_coalesced():
    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(
        (pd.DataFrame({
            "lower": ['one', None, 'one', 'four', None],
            "value": [1, 2, 3, 4, 5],
            "extra": [None, None, None, 'FO+ UR', None],
            "lowerNulls": ['one', 'two', None, 'four', None],
            "upperNulls": ["ONE", 'TWO', None, 'FOU  R', None]
        })))

    assert (coalesced(df)
            .where(F.col("coalescedCol").isNull())
            .count() == 0)


###############################################################################

def test_split():
    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(
        (pd.DataFrame({
            "before": ['a_b_c_d', None],
            "after": [['a', 'b', 'c', 'd'], None]
        })))

    assert (split(df, "before", col_out="new", split_on='_')
            .where((F.col("new") == F.col("after"))
            | (F.col("new").isNull() & F.col("after").isNull()))
            .count()
            ) == 2

    assert (split(df, "before", col_out=None, split_on='_')
            .where((F.col("before") == F.col("after"))
            | (F.col("before").isNull() & F.col("after").isNull()))
            .count()
            ) == 2


###############################################################################

def index_select_testing():
    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(
        (pd.DataFrame({
            "before": [['a', 'b', 'c'], None, ['b', 'c', 'd']],
            "after": ['a', None, 'b'],
            "afterneg": ['c', None, 'd']
        })))

    assert (index_select(df, "before", "test", 0)
            .where((F.col("test") == F.col("after"))
            | (F.col("test").isNull() & F.col("after").isNull()))
            .count()
            ) == 3

    assert (index_select(df, "before", "test", -1)
            .where((F.col("test") == F.col("afterneg"))
            | (F.col("test").isNull() & F.col("afterneg").isNull()))
            .count()
            ) == 3

###############################################################################


def test_clone_column():
    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(
        (pd.DataFrame({
            "UPPER": ["ONE£", 'TW""O', "T^^HREE", 'FO+UR', "FI@VE"],
        })))

    df2 = spark.createDataFrame(
        (pd.DataFrame({
            "NEW": ["ONE£", 'TW""O', "T^^HREE", 'FO+UR', "FI@VE"],
            "UPPER": ["ONE£", 'TW""O', "T^^HREE", 'FO+UR', "FI@VE"]
        })))
    # assert throwing error?
    # assert(clone_column(df,"NEW","UPPER").select("NEW")==df2.select("NEW"))
    assert (clone_column(df, "NEW", "UPPER")
            .where(F.col("NEW") == F.col("UPPER")).count() == 5)

    #######################################################################


def test_substring():
    spark = SparkSession.builder.getOrCreate()
    df = spark.createDataFrame(
        (pd.DataFrame({
            "NEW": ["ONE", 'TWO', 'THREE', 'FOUR'],
            "start": ["ONE", 'TWO', 'THR', 'FOU'],
            "end": ["ENO", "OWT", "EER", "RUO"]
        })))

    assert (substring(df, "final", "NEW", 1, 3)
            .where(F.col("start") == F.col("final")).count() == 4)

    assert (substring(df, "final", "NEW", 1, 3, True)
            .where(F.col("end") == F.col("final")).count() == 4)

##############################################################


def test_filter_window():

    spark = SparkSession.builder.getOrCreate()

    df = spark.createDataFrame(
        (pd.DataFrame({
            "col1": ['a', 'b', 'c', 'c', 'd', 'e', 'd'],
            "col2": [1, 1, 2, 2, 1, 1, 1]
        })))

    assert filter_window(df, 'col1', 'col2', 'count',
                         value=1, condition=True).count() == 3

    assert filter_window(df, 'col1', 'col2', 'count',
                         value=1, condition=False).count() == 4

    df = spark.createDataFrame(
        (pd.DataFrame({
            "col1": ['a', 'b', 'c', 'c', 'd', 'e', 'd'],
            "col2": [1, 1, 2, 3, 1, 1, 2]
        })))

    assert filter_window(df, 'col1', 'col2', 'countDistinct',
                         value=1, condition=True).count() == 3

    assert filter_window(df, 'col1', 'col2', 'count',
                         value=1, condition=False).count() == 4

    assert sorted(filter_window(df, 'col1', 'col2', 'min',
                                condition=True).toPandas()['col1']) ==\
        ['a', 'b', 'c', 'd', 'e']

    assert sorted(filter_window(df, 'col1', 'col2', 'min',
                                condition=True).toPandas()['col2']) ==\
        ['1', '1', '1', '1', '2']

    assert sorted(filter_window(df, 'col1', 'col2', 'min',
                                condition=False).toPandas()['col1']) ==\
        ['c', 'd']

    assert sorted(filter_window(df, 'col1', 'col2', 'min',
                                condition=False).toPandas()['col2']) ==\
        ['2', '3']

    assert sorted(filter_window(df, 'col1', 'col2', 'max',
                                condition=True).toPandas()['col1']) ==\
        ['a', 'b', 'c', 'd', 'e']

    assert sorted(filter_window(df, 'col1', 'col2', 'max',
                                condition=True).toPandas()['col2']) ==\
        ['1', '1', '1', '2', '3']

    assert sorted(filter_window(df, 'col1', 'col2', 'max',
                                condition=False).toPandas()['col1']) ==\
        ['c', 'd']

    assert sorted(filter_window(df, 'col1', 'col2', 'max',
                                condition=False).toPandas()['col2']) ==\
        ['1', '2']
