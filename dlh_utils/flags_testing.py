import pyspark
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import *
import pandas as pd
from dlh_utils.flags import *
from dlh_utils.dataframes import *
import chispa
from chispa import assert_df_equality
import pytest

pytestmark = pytest.mark.usefixtures("spark")


@pytest.fixture(scope="session")
def spark(request):
    """ fixture for creating a spark context
    Args:
        request: pytest.FixtureRequest object
    """
    spark = (SparkSession.builder.appName("flags_testing")
             .config('spark.executor.memory', '5g')
             .config('spark.yarn.excecutor.memoryOverhead', '2g')
             .getOrCreate())
    request.addfinalizer(lambda: spark.stop())
    return spark

###################################################################


class TestFlag1(object):
    def test_expected(self, spark):

        test_df = spark.createDataFrame(
            (pd.DataFrame({
                "ref_col": [x for x in range(40)] + [None]*10,
                "condition_col": 25})))

        result_df = flag(test_df,
                         ref_col='ref_col',
                         condition='==',
                         condition_value=25,
                         condition_col=None,
                         alias='test',
                         prefix='FLAG',
                         fill_null=None)

        intended_df = spark.createDataFrame(
            (pd.DataFrame({
                          "ref_col": [x for x in range(40)] + [None]*10,
                          "condition_col": 25,
                          "test": [False]*25
                          + [True]
                          + [False]*24})))

        assert_df_equality(intended_df,
                           result_df,
                           allow_nan_equality=True)

        assert_df_equality(intended_df,
                           result_df,
                           allow_nan_equality=True)

# ===================================================================


class TestFlag2(object):
    def test_expected(self, spark):

        test_df = spark.createDataFrame(
            (pd.DataFrame({
                "ref_col": [x for x in range(40)] + [None]*10,
                "condition_col": 25})))

        result_df = flag(test_df,
                         ref_col='ref_col',
                         condition='==',
                         condition_value=None,
                         condition_col='condition_col',
                         alias='test',
                         prefix='FLAG',
                         fill_null=None)

        intended_df = spark.createDataFrame(
            (pd.DataFrame({
                          "ref_col": [x for x in range(40)] + [None]*10,
                          "condition_col": 25,
                          "test": [False]*25
                          + [True]
                          + [False]*24})))

        assert_df_equality(intended_df,
                           result_df,
                           allow_nan_equality=True)

# ===================================================================


class TestFlag3(object):
    def test_expected(self, spark):

        test_df = spark.createDataFrame(
            (pd.DataFrame({
                "ref_col": [x for x in range(40)] + [None]*10,
                "condition_col": 25})))

        result_df = flag(test_df,
                         ref_col='ref_col',
                         condition='!=',
                         condition_value=25,
                         condition_col=None,
                         alias='test',
                         prefix='FLAG',
                         fill_null=None)

        intended_df = spark.createDataFrame(
            (pd.DataFrame({
                          "ref_col": [x for x in range(40)] + [None]*10,
                          "condition_col": 25,
                          "test": [True]*25
                          + [False]
                          + [True]*24})))

        assert_df_equality(result_df,
                           intended_df,
                           allow_nan_equality=True)

# ===================================================================


class TestFlag4(object):
    def test_expected(self, spark):

        test_df = spark.createDataFrame(
            (pd.DataFrame({
                "ref_col": [x for x in range(40)] + [None]*10,
                "condition_col": 25
            })))

        result_df = flag(test_df,
                         ref_col='ref_col',
                         condition='!=',
                         condition_value=None,
                         condition_col='condition_col',
                         alias='test',
                         prefix='FLAG',
                         fill_null=None)

        intended_df = spark.createDataFrame(
            (pd.DataFrame({
                "ref_col": [x for x in range(40)] + [None]*10,
                "condition_col": 25,
                "test": [True]*25
                + [False]
                + [True]*24})))

        assert_df_equality(result_df,
                           intended_df,
                           allow_nan_equality=True)

# ===================================================================


class TestFlag5(object):
    def test_expected(self, spark):

        test_df = spark.createDataFrame(
            (pd.DataFrame({
                "ref_col": [x for x in range(40)] + [None]*10,
                "condition_col": 25
            })))

        result_df = flag(test_df,
                         ref_col='ref_col',
                         condition='isNull',
                         condition_value=None,
                         condition_col=None,
                         alias='test',
                         prefix='FLAG',
                         fill_null=None)

        intended_df = spark.createDataFrame(
            (pd.DataFrame({
                "ref_col": [x for x in range(40)] + [None]*10,
                "condition_col": 25,
                "test": [False]*40
                + [True]*10})))

        assert_df_equality(result_df,
                           intended_df,
                           allow_nan_equality=True,
                           ignore_nullable=True)

# ===================================================================


class TestFlag6(object):
    def test_expected(self, spark):

        test_df = spark.createDataFrame(
            (pd.DataFrame({
                "ref_col": [x for x in range(40)] + [None]*10,
                "condition_col": 25
            })))

        result_df = flag(test_df,
                         ref_col='ref_col',
                         condition='isNotNull',
                         condition_value=None,
                         condition_col=None,
                         alias='test',
                         prefix='FLAG',
                         fill_null=None)

        intended_df = spark.createDataFrame(
            (pd.DataFrame({
                "ref_col": [x for x in range(40)] + [None]*10,
                "condition_col": 25,
                "test": [True]*40
                + [False]*10})))

        assert_df_equality(result_df,
                           intended_df,
                           allow_nan_equality=True,
                           ignore_nullable=True)

###################################################################


class TestFlagSummary1(object):
    def test_expected(self, spark):

        test_df = spark.createDataFrame(
            (pd.DataFrame({
                "ref_col": [x for x in range(40)] + [None]*10,
                "condition_col": 25
            })))

        result_df = flag(test_df,
                         ref_col='ref_col',
                         condition='==',
                         condition_value=25,
                         condition_col=None,
                         alias=None,
                         prefix='FLAG',
                         fill_null=None)

        intended_df = spark.createDataFrame(
            (pd.DataFrame({
                          "ref_col": [x for x in range(40)] + [None]*10,
                          "condition_col": 25,
                          "FLAG_ref_col==25": [False]*25
                          + [True]
                          + [False]*24})))

        assert_df_equality(result_df,
                           intended_df,
                           allow_nan_equality=True,
                           ignore_nullable=True,
                           ignore_column_order=True)

# ===================================================================


class TestFlagSummary2(object):
    def test_expected(self, spark):

        test_df = spark.createDataFrame(
            (pd.DataFrame({
                "ref_col": [x for x in range(40)] + [None]*10,
                "condition_col": 25
            })))

        result_df = flag(test_df,
                         ref_col='ref_col',
                         condition='!=',
                         condition_value=25,
                         condition_col=None,
                         alias=None,
                         prefix='FLAG',
                         fill_null=None)

        intended_df = spark.createDataFrame(
            (pd.DataFrame({
                "ref_col": [x for x in range(40)] + [None]*10,
                "condition_col": 25,
                "FLAG_ref_col!=25": [True]*25
                + [False]
                + [True]*24})))

        assert_df_equality(result_df,
                           intended_df,
                           allow_nan_equality=True,
                           ignore_nullable=True,
                           ignore_column_order=True)

# ===================================================================


class TestFlagSummary3(object):
    def test_expected(self, spark):

        test_df = spark.createDataFrame(
            (pd.DataFrame({
                "ref_col": [x for x in range(40)] + [None]*10,
                "condition_col": 25
            })))

        result_df = flag(test_df,
                         ref_col='ref_col',
                         condition='>=',
                         condition_value=25,
                         condition_col=None,
                         alias=None,
                         prefix='FLAG',
                         fill_null=None)

        intended_df = spark.createDataFrame(
            (pd.DataFrame({
                          "ref_col": [x for x in range(40)] + [None]*10,
                          "condition_col": 25,
                          "FLAG_ref_col>=25": [False]*25
                          + [True]*25})))

        assert_df_equality(result_df,
                           intended_df,
                           allow_nan_equality=True,
                           ignore_nullable=True,
                           ignore_column_order=True)

# ===================================================================


class TestFlagSummary4(object):
    def test_expected(self, spark):

        test_df = spark.createDataFrame(
            (pd.DataFrame({
                "ref_col": [x for x in range(40)] + [None]*10,
                "condition_col": 25
            })))

        result_df = flag(test_df,
                         ref_col='ref_col',
                         condition='<=',
                         condition_value=25,
                         condition_col=None,
                         alias=None,
                         prefix='FLAG',
                         fill_null=None)

        intended_df = spark.createDataFrame(
            (pd.DataFrame({
                          "ref_col": [x for x in range(40)] + [None]*10,
                          "condition_col": 25,
                          "FLAG_ref_col<=25": [True]*26
                          + [False]*24})))

        assert_df_equality(result_df,
                           intended_df,
                           allow_nan_equality=True,
                           ignore_nullable=True,
                           ignore_column_order=True)

# ===================================================================


class TestFlagSummary5(object):
    def test_expected(self, spark):

        test_df = spark.createDataFrame(
            (pd.DataFrame({
                "ref_col": [x for x in range(40)]+[None]*10,
                "condition_col": 25
            })))

        result_df = flag(test_df,
                         ref_col='ref_col',
                         condition='isNull',
                         condition_value=25,
                         condition_col=None,
                         alias=None,
                         prefix='FLAG',
                         fill_null=None)

        intended_df = spark.createDataFrame(
            (pd.DataFrame({
                "ref_col": [x for x in range(40)] + [None]*10,
                "condition_col": 25,
                "FLAG_ref_colisNull25": [False]*40
                + [True]*10})))

        assert_df_equality(result_df,
                           intended_df,
                           allow_nan_equality=True,
                           ignore_nullable=True,
                           ignore_column_order=True)

# ===================================================================


class TestFlagSummary6(object):
    def test_expected(self, spark):

        test_df = spark.createDataFrame(
            (pd.DataFrame({
                "ref_col": [x for x in range(40)]+[None]*10,
                "condition_col": 25
            })))

        result_df = flag(test_df,
                         ref_col='ref_col',
                         condition='isNotNull',
                         condition_value=25,
                         condition_col=None,
                         alias=None,
                         prefix='FLAG',
                         fill_null=None)

        intended_df = spark.createDataFrame(
            (pd.DataFrame({
                "ref_col": [x for x in range(40)] + [None]*10,
                "condition_col": 25,
                "FLAG_ref_colisNotNull25": [True]*40
                + [False]*10})))

        assert_df_equality(result_df,
                           intended_df,
                           allow_nan_equality=True,
                           ignore_nullable=True,
                           ignore_column_order=True)


###################################################################
class TestFlagCheck1(object):
    def test_expected(self, spark):

        test_df = spark.createDataFrame(
            (pd.DataFrame({
                "FLAG_1": ([True]*50)+([False]*50)})))

        result_df = flag_check(test_df,
                               prefix='FLAG_',
                               flags=None,
                               mode='master',
                               summary=False)

        intended_df1 = spark.createDataFrame(
            (pd.DataFrame({
                "FLAG_1": ([True]*50),
                'flag_count': [1]*50,
                'FAIL': [True]*50})))

        intended_df2 = spark.createDataFrame(
            (pd.DataFrame({
                "FLAG_1": ([False]*50),
                'flag_count': [0]*50,
                'FAIL': [False]*50})))

        intended_df = intended_df1.unionAll(intended_df2)

        assert_df_equality(result_df,
                           intended_df,
                           ignore_nullable=True,
                           ignore_column_order=True,
                           ignore_schema=True)

# ===================================================================


class TestFlagCheck2(object):
    def test_expected(self, spark):

        test_df = spark.createDataFrame(
            (pd.DataFrame({
                "FLAG_1": ([True]*50)+([False]*50)})))

        result_df = flag_check(test_df,
                               prefix='FLAG_',
                               flags=None,
                               mode='pass',
                               summary=False)

        intended_df = spark.createDataFrame((pd.DataFrame({
            "FLAG_1": ([False]*50),
            'flag_count': [0]*50,
            'FAIL': [False]*50})))

        assert_df_equality(result_df,
                           intended_df,
                           ignore_nullable=True,
                           ignore_column_order=True,
                           ignore_schema=True)

# ===================================================================


class TestFlagCheck3(object):
    def test_expected(self, spark):

        test_df = spark.createDataFrame(
            (pd.DataFrame({
                "FLAG_1": ([True]*50)+([False]*50)})))

        result_df = flag_check(test_df,
                               prefix='FLAG_',
                               flags=None,
                               mode='fail',
                               summary=False)

        intended_df = spark.createDataFrame((pd.DataFrame({
            "FLAG_1": ([True]*50),
            'flag_count': [1]*50,
            'FAIL': [True]*50})))

        assert_df_equality(result_df,
                           intended_df,
                           ignore_nullable=True,
                           ignore_column_order=True,
                           ignore_schema=True)

# ===================================================================


class TestFlagCheck4(object):
    def test_expected(self, spark):

        test_df = spark.createDataFrame(
            (pd.DataFrame({"FLAG_1": [True]*50
                           + [False]*50})))

        result_df1, result_df2 = flag_check(test_df,
                                            prefix='FLAG_',
                                            flags=None,
                                            mode='split',
                                            summary=False)

        intended_df = spark.createDataFrame(
            (pd.DataFrame({
                "FLAG_1": ([True]*50),
                          "flag_count": [1]*50,
                          "FAIL": [True]*50})))

        assert_df_equality(result_df2,
                           intended_df,
                           ignore_nullable=True,
                           ignore_column_order=True,
                           ignore_schema=True)

# ===================================================================


class TestFlagCheck5(object):
    def test_expected(self, spark):

        test_df = spark.createDataFrame(
            (pd.DataFrame({"FLAG_1": [True]*50
                           + [False]*50})))

        result_df1, result_df2 = flag_check(test_df,
                                            prefix='FLAG_',
                                            flags=None,
                                            mode='master',
                                            summary=True)

        intended_df1 = spark.createDataFrame(
            pd.concat([pd.DataFrame({
                "FLAG_1": [True]*50,
                'flag_count': [1]*50,
                'FAIL': [True]*50}),
                pd.DataFrame({
                    "FLAG_1": [False]*50,
                    'flag_count': [0]*50,
                    'FAIL': [False]*50})]
            ))

        intended_df2 = spark.createDataFrame(
            (pd.DataFrame({
                "flag": ['FLAG_1', 'FAIL'],
                'true': [50]*2,
                'false': [50]*2,
                'rows': [100]*2,
                'percent_true': [50.0]*2,
                'percent_false': [50.0]*2
            })))

        assert_df_equality(result_df1,
                           intended_df1,
                           ignore_nullable=True,
                           ignore_column_order=True,
                           ignore_schema=True)

        assert_df_equality(result_df2,
                           intended_df2,
                           ignore_nullable=True,
                           ignore_column_order=True,
                           ignore_schema=True)

# ===================================================================


class TestFlagCheck6(object):
    def test_expected(self, spark):

        pretest_df_orig = spark.createDataFrame(
            (pd.DataFrame({"FLAG_1": [True]*50
                           + [False]*50})))

        pretest_df1, pretest_df2 = flag_check(pretest_df_orig,
                                              prefix='FLAG_',
                                              flags=None,
                                              mode='master',
                                              summary=True)

        test_df = pretest_df2.toPandas() == \
            spark.createDataFrame(
            [('FLAG_1', 50, 50, 100, 50.0, 50.0),
             ('FAIL', 50, 50, 100, 50.0, 50.0)],
            ['flag', 'true', 'false', 'rows',
             'percent_true', 'percent_false']
        ).toPandas()

        result_df = spark.createDataFrame(test_df)

        intended_df = spark.createDataFrame(pd.DataFrame({
            "flag": [True]*2,
            'true': [True]*2,
            'false': [True]*2,
            'rows': [True]*2,
            'percent_true': [True]*2,
            'percent_false': [True]*2
        }))

        assert_df_equality(result_df,
                           intended_df,
                           ignore_nullable=True,
                           ignore_column_order=True,
                           ignore_schema=True)

###################################################################
# END
