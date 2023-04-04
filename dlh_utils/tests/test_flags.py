'''
Pytesting on Dataframes functions
'''
import pandas as pd
import pytest
from chispa import assert_df_equality
from dlh_utils.flags import flag, flag_check

pytestmark = pytest.mark.usefixtures("spark")

###################################################################
class TestFlag1:
    """Test for flag function"""

    @staticmethod
    def test_expected1(spark):
        """Test the expected functionality"""

        test_df = spark.createDataFrame((pd.DataFrame(
            {"ref_col": list(range(40)) + [None] * 10, "condition_col": 25})))

        result_df = flag(
            test_df,
            ref_col="ref_col",
            condition="==",
            condition_value=25,
            condition_col=None,
            alias="test",
            prefix="FLAG",
            fill_null=None,
        )

        intended_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "ref_col": list(range(40)) + [None] * 10,
                        "condition_col": 25,
                        "test": [False] * 25 + [True] + [False] * 24,
                    }
                )
            )
        )

        assert_df_equality(intended_df, result_df, allow_nan_equality=True)

# ===================================================================
    @staticmethod
    def test_expected2(spark):
        """Test the expected functionality"""

        test_df = spark.createDataFrame((pd.DataFrame(
            {"ref_col": list(range(40)) + [None] * 10, "condition_col": 25})))

        result_df = flag(
            test_df,
            ref_col="ref_col",
            condition="==",
            condition_value=None,
            condition_col="condition_col",
            alias="test",
            prefix="FLAG",
            fill_null=None,
        )

        intended_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "ref_col": list(range(40)) + [None] * 10,
                        "condition_col": 25,
                        "test": [False] * 25 + [True] + [False] * 24,
                    }
                )
            )
        )

        assert_df_equality(intended_df, result_df, allow_nan_equality=True)


# ===================================================================
    @staticmethod
    def test_expected3(spark):
        """Test the expected functionality"""

        test_df = spark.createDataFrame((pd.DataFrame(
            {"ref_col": list(range(40)) + [None] * 10, "condition_col": 25})))

        result_df = flag(
            test_df,
            ref_col="ref_col",
            condition="!=",
            condition_value=25,
            condition_col=None,
            alias="test",
            prefix="FLAG",
            fill_null=None,
        )

        intended_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "ref_col": list(range(40)) + [None] * 10,
                        "condition_col": 25,
                        "test": [True] * 25 + [False] + [True] * 24,
                    }
                )
            )
        )

        assert_df_equality(result_df, intended_df, allow_nan_equality=True)


# ===================================================================
    @staticmethod
    def test_expected4(spark):
        """Test the expected functionality"""

        test_df = spark.createDataFrame((pd.DataFrame(
            {"ref_col": list(range(40)) + [None] * 10, "condition_col": 25})))

        result_df = flag(
            test_df,
            ref_col="ref_col",
            condition="!=",
            condition_value=None,
            condition_col="condition_col",
            alias="test",
            prefix="FLAG",
            fill_null=None,
        )

        intended_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "ref_col": list(range(40)) + [None] * 10,
                        "condition_col": 25,
                        "test": [True] * 25 + [False] + [True] * 24,
                    }
                )
            )
        )

        assert_df_equality(result_df, intended_df, allow_nan_equality=True)


# ===================================================================
    @staticmethod
    def test_expected5(spark):
        """Test the expected functionality"""

        test_df = spark.createDataFrame((pd.DataFrame(
            {"ref_col": list(range(40)) + [None] * 10, "condition_col": 25})))

        result_df = flag(
            test_df,
            ref_col="ref_col",
            condition="isNull",
            condition_value=None,
            condition_col=None,
            alias="test",
            prefix="FLAG",
            fill_null=None,
        )

        intended_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "ref_col": list(range(40)) + [None] * 10,
                        "condition_col": 25,
                        "test": [False] * 40 + [True] * 10,
                    }
                )
            )
        )

        assert_df_equality(
            result_df,
            intended_df,
            allow_nan_equality=True,
            ignore_nullable=True)


# ===================================================================
    @staticmethod
    def test_expected6(spark):
        """Test the expected functionality"""

        test_df = spark.createDataFrame((pd.DataFrame(
            {"ref_col": list(range(40)) + [None] * 10, "condition_col": 25})))

        result_df = flag(
            test_df,
            ref_col="ref_col",
            condition="isNotNull",
            condition_value=None,
            condition_col=None,
            alias="test",
            prefix="FLAG",
            fill_null=None,
        )

        intended_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "ref_col": list(range(40)) + [None] * 10,
                        "condition_col": 25,
                        "test": [True] * 40 + [False] * 10,
                    }
                )
            )
        )

        assert_df_equality(
            result_df,
            intended_df,
            allow_nan_equality=True,
            ignore_nullable=True)


###################################################################
class TestFlagSummary1:
    """Test for flag_summary function"""

    @staticmethod
    def test_expected7(spark):
        """Test the expected functionality"""

        test_df = spark.createDataFrame((pd.DataFrame(
            {"ref_col": list(range(40)) + [None] * 10, "condition_col": 25})))

        result_df = flag(
            test_df,
            ref_col="ref_col",
            condition="==",
            condition_value=25,
            condition_col=None,
            alias=None,
            prefix="FLAG",
            fill_null=None,
        )

        intended_df = spark.createDataFrame(
            (pd.DataFrame(
                {
                    "ref_col": list(range(40)) + [None] * 10,
                    "condition_col": 25,
                    "FLAG_ref_col==25": [False] * 25 + [True] + [False] * 24,
                })))

        assert_df_equality(
            result_df,
            intended_df,
            allow_nan_equality=True,
            ignore_nullable=True,
            ignore_column_order=True,
        )


# ===================================================================
    @staticmethod
    def test_expected8(spark):
        """Test the expected functionality"""

        test_df = spark.createDataFrame((pd.DataFrame(
            {"ref_col": list(range(40)) + [None] * 10,
             "condition_col": 25})))

        result_df = flag(
            test_df,
            ref_col="ref_col",
            condition="!=",
            condition_value=25,
            condition_col=None,
            alias=None,
            prefix="FLAG",
            fill_null=None,
        )

        intended_df = spark.createDataFrame(
            (pd.DataFrame(
                {
                    "ref_col": list(range(40)) +
                    [None] *
                    10,
                    "condition_col": 25,
                    "FLAG_ref_col!=25": [True] * 25 + [False] + [True] * 24,
                })))

        assert_df_equality(
            result_df,
            intended_df,
            allow_nan_equality=True,
            ignore_nullable=True,
            ignore_column_order=True,
        )


# ===================================================================
    @staticmethod
    def test_expected9(spark):
        """Test the expected functionality"""

        test_df = spark.createDataFrame((pd.DataFrame(
            {"ref_col": list(range(40)) + [None] * 10,
             "condition_col": 25})))

        result_df = flag(
            test_df,
            ref_col="ref_col",
            condition=">=",
            condition_value=25,
            condition_col=None,
            alias=None,
            prefix="FLAG",
            fill_null=None,
        )

        intended_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "ref_col": list(range(40)) + [None] * 10,
                        "condition_col": 25,
                        "FLAG_ref_col>=25": [False] * 25 + [True] * 25,
                    }
                )
            )
        )

        assert_df_equality(
            result_df,
            intended_df,
            allow_nan_equality=True,
            ignore_nullable=True,
            ignore_column_order=True,
        )


# ===================================================================
    @staticmethod
    def test_expected10(spark):
        """Test the expected functionality"""

        test_df = spark.createDataFrame((pd.DataFrame(
            {"ref_col": list(range(40)) + [None] * 10, "condition_col": 25})))

        result_df = flag(
            test_df,
            ref_col="ref_col",
            condition="<=",
            condition_value=25,
            condition_col=None,
            alias=None,
            prefix="FLAG",
            fill_null=None,
        )

        intended_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "ref_col": list(range(40)) + [None] * 10,
                        "condition_col": 25,
                        "FLAG_ref_col<=25": [True] * 26 + [False] * 24,
                    }
                )
            )
        )

        assert_df_equality(
            result_df,
            intended_df,
            allow_nan_equality=True,
            ignore_nullable=True,
            ignore_column_order=True,
        )


# ===================================================================
    @staticmethod
    def test_expected11(spark):
        """Test the expected functionality"""

        test_df = spark.createDataFrame((pd.DataFrame(
            {"ref_col": list(range(40)) + [None] * 10, "condition_col": 25})))

        result_df = flag(
            test_df,
            ref_col="ref_col",
            condition="isNull",
            condition_value=25,
            condition_col=None,
            alias=None,
            prefix="FLAG",
            fill_null=None,
        )

        intended_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "ref_col": list(range(40)) + [None] * 10,
                        "condition_col": 25,
                        "FLAG_ref_colisNull25": [False] * 40 + [True] * 10,
                    }
                )
            )
        )

        assert_df_equality(
            result_df,
            intended_df,
            allow_nan_equality=True,
            ignore_nullable=True,
            ignore_column_order=True,
        )


# ===================================================================
    @staticmethod
    def test_expected12(spark):
        """Test the expected functionality"""

        test_df = spark.createDataFrame((pd.DataFrame(
            {"ref_col": list(range(40)) + [None] * 10, "condition_col": 25})))

        result_df = flag(
            test_df,
            ref_col="ref_col",
            condition="isNotNull",
            condition_value=25,
            condition_col=None,
            alias=None,
            prefix="FLAG",
            fill_null=None,
        )

        intended_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "ref_col": list(range(40)) + [None] * 10,
                        "condition_col": 25,
                        "FLAG_ref_colisNotNull25": [True] * 40 + [False] * 10,
                    }
                )
            )
        )

        assert_df_equality(
            result_df,
            intended_df,
            allow_nan_equality=True,
            ignore_nullable=True,
            ignore_column_order=True,
        )
        
# ===================================================================
    @staticmethod
    def test_expected13(spark):
        """Test the expected functionality"""

        test_df = spark.createDataFrame((pd.DataFrame(
            {"ref_col": list(range(40)) + [None] * 10, "condition_col": 25})))

        result_df = flag(
            test_df,
            ref_col="ref_col",
            condition="regex",
            condition_value="^1",
            condition_col=None,
            alias="test",
            prefix="FLAG",
            fill_null=None)

        intended_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "ref_col": list(range(40)) + [None] * 10,
                        "condition_col": 25,
                        "test": [False] + [True] + ([False] * 8)
                                + ([True] * 10) + ([False] * 30)
                    }
                )
            )
        )

        assert_df_equality(intended_df, result_df, allow_nan_equality=True)


###################################################################
class TestFlagCheck1:
    """Test for flag_check function"""

    @staticmethod
    def test_expected14(spark):
        """Test the expected functionality"""

        test_df = spark.createDataFrame(
            (pd.DataFrame({"FLAG_1": ([True] * 50) + ([False] * 50)}))
        )

        result_df = flag_check(
            test_df, prefix="FLAG_", flags=None, mode="master", summary=False
        )

        intended_df1 = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "FLAG_1": ([True] * 50),
                        "flag_count": [1] * 50,
                        "FAIL": [True] * 50,
                    }
                )
            )
        )

        intended_df2 = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "FLAG_1": ([False] * 50),
                        "flag_count": [0] * 50,
                        "FAIL": [False] * 50,
                    }
                )
            )
        )

        intended_df = intended_df1.union(intended_df2)

        assert_df_equality(
            result_df,
            intended_df,
            ignore_nullable=True,
            ignore_column_order=True,
            ignore_schema=True,
        )


# ===================================================================
    @staticmethod
    def test_expected15(spark):
        """Test the expected functionality"""

        test_df = spark.createDataFrame(
            (pd.DataFrame({"FLAG_1": ([True] * 50) + ([False] * 50)}))
        )

        result_df = flag_check(
            test_df, prefix="FLAG_", flags=None, mode="pass", summary=False
        )

        intended_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "FLAG_1": ([False] * 50),
                        "flag_count": [0] * 50,
                        "FAIL": [False] * 50,
                    }
                )
            )
        )

        assert_df_equality(
            result_df,
            intended_df,
            ignore_nullable=True,
            ignore_column_order=True,
            ignore_schema=True,
        )


# ===================================================================
    @staticmethod
    def test_expected16(spark):
        """Test the expected functionality"""

        test_df = spark.createDataFrame(
            (pd.DataFrame({"FLAG_1": ([True] * 50) + ([False] * 50)}))
        )

        result_df = flag_check(
            test_df, prefix="FLAG_", flags=None, mode="fail", summary=False
        )

        intended_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "FLAG_1": ([True] * 50),
                        "flag_count": [1] * 50,
                        "FAIL": [True] * 50,
                    }
                )
            )
        )

        assert_df_equality(
            result_df,
            intended_df,
            ignore_nullable=True,
            ignore_column_order=True,
            ignore_schema=True,
        )


# ===================================================================
    @staticmethod
    def test_expected17(spark):
        """Test the expected functionality"""

        test_df = spark.createDataFrame(
            (pd.DataFrame({"FLAG_1": [True] * 50 + [False] * 50}))
        )

        result_df2 = flag_check(
            test_df, prefix="FLAG_", flags=None, mode="split", summary=False)[1]

        intended_df = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "FLAG_1": ([True] * 50),
                        "flag_count": [1] * 50,
                        "FAIL": [True] * 50,
                    }
                )
            )
        )

        assert_df_equality(
            result_df2,
            intended_df,
            ignore_nullable=True,
            ignore_column_order=True,
            ignore_schema=True,
        )


# ===================================================================
    @staticmethod
    def test_expected18(spark):
        """Test the expected functionality"""

        test_df = spark.createDataFrame(
            (pd.DataFrame({"FLAG_1": [True] * 50 + [False] * 50}))
        )

        result_df1, result_df2 = flag_check(
            test_df, prefix="FLAG_", flags=None, mode="master", summary=True
        )

        intended_df1 = spark.createDataFrame(
            pd.concat(
                [
                    pd.DataFrame(
                        {
                            "FLAG_1": [True] * 50,
                            "flag_count": [1] * 50,
                            "FAIL": [True] * 50,
                        }
                    ),
                    pd.DataFrame(
                        {
                            "FLAG_1": [False] * 50,
                            "flag_count": [0] * 50,
                            "FAIL": [False] * 50,
                        }
                    ),
                ]
            )
        )

        intended_df2 = spark.createDataFrame(
            (
                pd.DataFrame(
                    {
                        "flag": ["FLAG_1", "FAIL"],
                        "true": [50] * 2,
                        "false": [50] * 2,
                        "rows": [100] * 2,
                        "percent_true": [50.0] * 2,
                        "percent_false": [50.0] * 2,
                    }
                )
            )
        )

        assert_df_equality(
            result_df1,
            intended_df1,
            ignore_nullable=True,
            ignore_column_order=True,
            ignore_schema=True,
        )

        assert_df_equality(
            result_df2,
            intended_df2,
            ignore_nullable=True,
            ignore_column_order=True,
            ignore_schema=True,
        )


# ===================================================================
    @staticmethod
    def test_expected19(spark):
        """Test the expected functionality"""

        pretest_df_orig = spark.createDataFrame(
            (pd.DataFrame({"FLAG_1": [True] * 50 + [False] * 50}))
        )

        pretest_df2 = flag_check(
            pretest_df_orig, prefix="FLAG_", flags=None, mode="master", summary=True)[1]

        test_df = (
            pretest_df2.toPandas()
            == spark.createDataFrame(
                [
                    ("FLAG_1", 50, 50, 100, 50.0, 50.0),
                    ("FAIL", 50, 50, 100, 50.0, 50.0),
                ],
                ["flag", "true", "false", "rows", "percent_true", "percent_false"],
            ).toPandas()
        )

        result_df = spark.createDataFrame(test_df)

        intended_df = spark.createDataFrame(
            pd.DataFrame(
                {
                    "flag": [True] * 2,
                    "true": [True] * 2,
                    "false": [True] * 2,
                    "rows": [True] * 2,
                    "percent_true": [True] * 2,
                    "percent_false": [True] * 2,
                }
            )
        )

        assert_df_equality(
            result_df,
            intended_df,
            ignore_nullable=True,
            ignore_column_order=True,
            ignore_schema=True,
        )


###################################################################
