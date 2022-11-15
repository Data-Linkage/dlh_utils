import pyspark
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pandas as pd
from dlh_utils.flags import *

##################################################################


def test_flag():

    spark = SparkSession.builder.getOrCreate()

    df = spark.createDataFrame(
        (pd.DataFrame({
            "ref_col": [x for x in range(40)]+[None]*10,
            "condition_col": 25
        })))

    assert (flag(df,
                 ref_col='ref_col',
                 condition='==',
                 condition_value=25,
                 condition_col=None,
                 alias='test',
                 prefix='FLAG',
                 fill_null=None)
            .where(F.col('test') == True)
            ).count() == 1

    assert (flag(df,
                 ref_col='ref_col',
                 condition='==',
                 condition_value=None,
                 condition_col='condition_col',
                 alias='test',
                 prefix='FLAG',
                 fill_null=None)
            .where(F.col('test') == True)
            ).count() == 1

    assert (flag(df,
                 ref_col='ref_col',
                 condition='!=',
                 condition_value=25,
                 condition_col=None,
                 alias='test',
                 prefix='FLAG',
                 fill_null=None)
            .where(F.col('test') == True)
            ).count() == 49

    assert (flag(df,
                 ref_col='ref_col',
                 condition='!=',
                 condition_value=None,
                 condition_col='condition_col',
                 alias='test',
                 prefix='FLAG',
                 fill_null=None)
            .where(F.col('test') == True)
            ).count() == 49

    assert (flag(df,
                 ref_col='ref_col',
                 condition='isNull',
                 condition_value=None,
                 condition_col=None,
                 alias='test',
                 prefix='FLAG',
                 fill_null=None)
            .where(F.col('test') == True)
            ).count() == 10

    assert (flag(df,
                 ref_col='ref_col',
                 condition='isNotNull',
                 condition_value=None,
                 condition_col=None,
                 alias='test',
                 prefix='FLAG',
                 fill_null=None)
            .where(F.col('test') == True)
            ).count() == 40

##################################################################


def test_flag_summary():

    spark = SparkSession.builder.getOrCreate()

    df = spark.createDataFrame(
        (pd.DataFrame({
            "ref_col": [x for x in range(40)]+[None]*10,
            "condition_col": 25
        })))

    df = flag(df,
              ref_col='ref_col',
              condition='==',
              condition_value=25,
              condition_col=None,
              alias=None,
              prefix='FLAG',
              fill_null=None)

    df = flag(df,
              ref_col='ref_col',
              condition='!=',
              condition_value=25,
              condition_col=None,
              alias=None,
              prefix='FLAG',
              fill_null=None)

    df = flag(df,
              ref_col='ref_col',
              condition='>=',
              condition_value=25,
              condition_col=None,
              alias=None,
              prefix='FLAG',
              fill_null=None)

    df = flag(df,
              ref_col='ref_col',
              condition='<=',
              condition_value=25,
              condition_col=None,
              alias=None,
              prefix='FLAG',
              fill_null=None)

    df = flag(df,
              ref_col='ref_col',
              condition='isNull',
              condition_value=25,
              condition_col=None,
              alias=None,
              prefix='FLAG',
              fill_null=None)

    df = flag(df,
              ref_col='ref_col',
              condition='isNotNull',
              condition_value=25,
              condition_col=None,
              alias=None,
              prefix='FLAG',
              fill_null=None)

    assert list(flag_summary(df, [x for x in df.columns
                                  if x.startswith('FLAG_')],
                             pandas=True)['flag']) ==\
        ['FLAG_ref_col==25',
         'FLAG_ref_col!=25',
         'FLAG_ref_col>=25',
         'FLAG_ref_col<=25',
         'FLAG_ref_colisNull25',
         'FLAG_ref_colisNotNull25']

    assert list(flag_summary(df, [x for x in df.columns
                                  if x.startswith('FLAG_')],
                             pandas=True)['percent_true']) ==\
        [2.0, 98.0, 50.0, 52.0, 20.0, 80.0]

    assert list(flag_summary(df, [x for x in df.columns
                                  if x.startswith('FLAG_')],
                             pandas=True)['percent_false']) ==\
        [98.0, 2.0, 50.0, 48.0, 80.0, 20.0]

####################################################################


def test_flag_check():

    spark = SparkSession.builder.getOrCreate()

    df = spark.createDataFrame(
        (pd.DataFrame({
            "FLAG_1": ([True]*50)+([False]*50),
        })))

    master = flag_check(df,
                        prefix='FLAG_',
                        flags=None,
                        mode='master',
                        summary=False)

    assert master.count() == 100

    assert (master
            .where(F.col('FAIL') == True)
            ).count() == 50

    assert (master
            .where(F.col('FAIL') == False)
            ).count() == 50

    pass_df = flag_check(df,
                         prefix='FLAG_',
                         flags=None,
                         mode='pass',
                         summary=False)

    assert pass_df.count() == 50

    assert (pass_df
            .where(F.col('FAIL') == True)
            ).count() == 0

    assert (pass_df
            .where(F.col('FAIL') == False)
            ).count() == 50

    fail_df = flag_check(df,
                         prefix='FLAG_',
                         flags=None,
                         mode='fail',
                         summary=False)

    assert fail_df.count() == 50

    assert (fail_df
            .where(F.col('FAIL') == True)
            ).count() == 50

    assert (fail_df
            .where(F.col('FAIL') == False)
            ).count() == 0

    pass_df, fail_df = flag_check(df,
                                  prefix='FLAG_',
                                  flags=None,
                                  mode='split',
                                  summary=False)

    assert pass_df.count() == 50

    assert (pass_df
            .where(F.col('FAIL') == True)
            ).count() == 0

    assert (pass_df
            .where(F.col('FAIL') == False)
            ).count() == 50

    assert fail_df.count() == 50

    assert (fail_df
            .where(F.col('FAIL') == True)
            ).count() == 50

    assert (fail_df
            .where(F.col('FAIL') == False)
            ).count() == 0

    master, summary = flag_check(df,
                                 prefix='FLAG_',
                                 flags=None,
                                 mode='master',
                                 summary=True)

    summary = summary.toPandas() ==\
        spark.createDataFrame(
        [('FLAG_1', 50, 50, 100, 50.0, 50.0),
         ('FAIL', 50, 50, 100, 50.0, 50.0)],
        ['flag', 'true', 'false', 'rows', 'percent_true', 'percent_false']
    ).toPandas()

    assert [summary[summary[variable] != True].shape[0]
            for variable in list(summary)] == [0]*len(list(summary))
