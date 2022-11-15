'''
Flag functions, used to quickly highlight anomalous values within data
'''
from operator import add
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType
import pyspark.sql.functions as F
import pandas as pd
from functools import reduce

###############################################################################


def flag(df, ref_col, condition, condition_value=None, condition_col=None,
         alias=None, prefix='FLAG', fill_null=None):
    """
    Adds True or False flags to supplied dataframe that can then be used for
    quality checks.

    Conditions can be set in comparison to columns or specific values
    (e.g. == column, ==1).  Conditions covered are equals, not equals,
    greater/less than, is/is not null. Optional TRUE/FALSE
    fill for null outputs of comparision. Designed for use in conjunction with
    flag_summary() and flag_check() functions.

    This function creates a column filled with TRUE or FALSE values
    based on whether a condition has been met.

    NOTE: If an alias is not specified, a Flag column name is automatically
    generated.

    Parameters
    ----------
    df : dataframe
      The dataframe the function is applied to.
    ref_col : string
      The column title that the conditions are
      performing checks upon.
    condition : {'==','!=','>','>=',<=','<','isNull','isNotNull'}
      Conditional statements used to compare values to the
      ref_col.
    condition_value : data-types, default = None
      The value the ref_col is being compared against.
    condition_col : data-types, default = None
      Comparison column for flag condition
    alias : string, default = None
      Alias for flag column.
    prefix : string, default = 'FLAG'
      Default alias flag column prefix.
    fill_null : bool, default = False
      True or False fill where condition operations
      return null.

    Returns
    -------
    dataframe
      Dataframe with additional window column.

    Raises
    ------
    None at present.

    Example
    -------

    > df.show()

    +---+--------+----------+-------+----------+---+--------+
    | ID|Forename|Middlename|Surname|       DoB|Sex|Postcode|
    +---+--------+----------+-------+----------+---+--------+
    |  1|   Homer|       Jay|Simpson|1983-05-12|  M|ET74 2SP|
    |  2|   Marge|    Juliet|Simpson|1983-03-19|  F|ET74 2SP|
    |  3|    Bart|     Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|
    |  3|    Bart|     Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|
    |  4|    Lisa|     Marie|Simpson|2014-05-09|  F|ET74 2SP|
    |  5|  Maggie|      null|Simpson|2021-01-12|  F|ET74 2SP|
    +---+--------+----------+-------+----------+---+--------+

    > flag(df,
           ref_col = 'Middlename',
           condition = 'isNotNull',
           condition_value=None,
           condition_col=None,
           alias = None,
           prefix='FLAG',
           fill_null = None).show()

    +---+--------+----------+-------+----------+---+--------+------------------------+
    | ID|Forename|Middlename|Surname|       DoB|Sex|Postcode|FLAG_MiddlenameisNotNull|
    +---+--------+----------+-------+----------+---+--------+------------------------+
    |  1|   Homer|       Jay|Simpson|1983-05-12|  M|ET74 2SP|                    true|
    |  2|   Marge|    Juliet|Simpson|1983-03-19|  F|ET74 2SP|                    true|
    |  3|    Bart|     Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|                    true|
    |  3|    Bart|     Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|                    true|
    |  4|    Lisa|     Marie|Simpson|2014-05-09|  F|ET74 2SP|                    true|
    |  5|  Maggie|      null|Simpson|2021-01-12|  F|ET74 2SP|                   false|
    +---+--------+----------+-------+----------+---+--------+------------------------+
    """

    if (alias is None
            and condition_value is not None):

        alias_value = str(condition_value)
        alias = f"{prefix}_{ref_col}{condition}{alias_value}"

    if (alias is None
            and condition_col is not None):
        alias = f"{prefix}_{ref_col}{condition}_{condition_col}"

    if (alias is None
        and condition_col is None
            and condition_value is None):
        alias = f"{prefix}_{ref_col}{condition}"

    if (condition == '=='
            and condition_col is not None):
        df = df.withColumn(alias,
                           F.col(ref_col) == F.col(condition_col))

    if (condition == '>'
            and condition_col is not None):
        df = df.withColumn(alias,
                           F.col(ref_col) > F.col(condition_col))

    if (condition == '>='
            and condition_col is not None):
        df = df.withColumn(alias,
                           F.col(ref_col) >= F.col(condition_col))

    if (condition == '<'
            and condition_col is not None):
        df = df.withColumn(alias,
                           F.col(ref_col) < F.col(condition_col))

    if (condition == '<='
            and condition_col is not None):
        df = df.withColumn(alias,
                           F.col(ref_col) <= F.col(condition_col))

    if (condition == '!='
            and condition_col is not None):
        df = df.withColumn(alias,
                           F.col(ref_col) != F.col(condition_col))

    if condition == 'isNull':
        df = df.withColumn(alias,
                           (F.col(ref_col).isNull()) | (
                               F.isnan(F.col(ref_col)))
                           )

    if condition == 'isNotNull':
        df = df.withColumn(alias,
                           (F.col(ref_col).isNotNull()) & (
                               F.isnan(F.col(ref_col)) == False)
                           )

    if fill_null is not None:
        df = (df
              .withColumn(alias,
                          F.when(F.col(alias).isNull(),
                                 fill_null)
                          .otherwise(F.col(alias)))
              )

    return df
###############################################################################

# Potential to imporve with automated identifiaction of flag columns
# e.g. through prefix or regex - as used in flag_check())


def flag_summary(df, flags=None, pandas=False):
    """
    Produces summary table of boolean flag columns.

    Produces a summary of True/False counts and percentages.
    Option to output as pandas or spark dataframe (default
    spark).

    Parameters
    ----------
    df : dataframe
      The dataframe the function is applied to.
    flags : string or list of strings
      A boolean flag column title in the format
      of a string or a list of strings of boolean
      flag column titles.
    pandas : bool, default = False
      Option to output as a pandas dataframe.

    Returns
    -------
    dataframe
      Dataframe with additional window column.

    Raises
    ------
    None at present

    Example
    -------

    > df.show()
    +---+--------+----------+-------+----------+---+--------+------------------------+
    | ID|Forename|Middlename|Surname|       DoB|Sex|Postcode|FLAG_MiddlenameisNotNull|
    +---+--------+----------+-------+----------+---+--------+------------------------+
    |  1|   Homer|       Jay|Simpson|1983-05-12|  M|ET74 2SP|                    true|
    |  2|   Marge|    Juliet|Simpson|1983-03-19|  F|ET74 2SP|                    true|
    |  3|    Bart|     Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|                    true|
    |  3|    Bart|     Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|                    true|
    |  4|    Lisa|     Marie|Simpson|2014-05-09|  F|ET74 2SP|                    true|
    |  5|  Maggie|      null|Simpson|2021-01-12|  F|ET74 2SP|                   false|
    +---+--------+----------+-------+----------+---+--------+------------------------+

    > flag_summary(df,flags = None,pandas=False).show()

    +--------------------+----+-----+----+-----------------+------------------+
    |                flag|true|false|rows|     percent_true|     percent_false|
    +--------------------+----+-----+----+-----------------+------------------+
    |FLAG_Middlenameis...|   5|    1|   6|83.33333333333334|16.666666666666657|
    +--------------------+----+-----+----+-----------------+------------------+

    """
    spark = SparkSession.builder.getOrCreate()

    if flags is None:
        flags = [
            column for column in df.columns if column.startswith('FLAG_')]

    if type(flags) != list:
        flags = [flags]

    rows = df.count()

    flags_out = []

    for col in flags:

        flags_out.append((df
                          .select(col)
                         .where(F.col(col) == True)

                          .count()
                          ))

    out = pd.DataFrame({
        'flag': flags,
        'true': flags_out,
        'false': [rows-x for x in flags_out],
        'rows': rows,
        'percent_true': [(x/rows)*100 for x in flags_out],
        'percent_false': [100-((x/rows)*100) for x in flags_out]
    })

    out = out[[
        'flag',
        'true',
        'false',
        'rows',
        'percent_true',
        'percent_false'
    ]]

    if pandas == False:

        out = (spark
               .createDataFrame(out)
               .coalesce(1))

    return out

###############################################################################


def flag_check(df, prefix='FLAG_', flags=None,  mode='master', summary=False):
    """
    Reads flag columns and counts True/ Fail values.

    Adds flag count column (counting TRUE/Fail values) and overall
    fail column (TRUE/FALSE and flag TRUE/Fail). If any rows in the flag count
    column are greater than 0, the overall fail value for this row will be True so
    this is quickly highlighted to the user.

    Option to produce flag summary
    stats employing flag_summary(). Option to return full dataframe, only passes, 
    only fails, or passes and fails(residuals) as two separate dataframes.

    Parameters
    ----------
    df : dataframe
      The dataframe the function is applied to.
    prefix : string, default = 'FLAG_'
      For dynamic identification of flag columns if prefixed.
    flags : list of strings, default = None
      List of flag manually specified flags to operate the function
      on. If this is kept as default value, all columns in df
      that start with 'FLAG_' are assumed to be flag columns by the
      function.
    mode : {'master','split','pass','fail'}
      master: returns all results (full dataframe).
      pass: only returns rows where pass is True.
      fail: only returns rows where fail is True.
      split: returns two seperate dataframes for both
      pass and fail results.
    summary : bool, default = False
      optional flag summary employing flag_summary() function

    Returns
    -------
    dataframe
      Returns dataframe with results depending on the
      mode argument.
      If the mode argument is set to split, it will return
      two dataframes.

    Raises
    ------
    None at present

    Example
    -------

    > df.show()
    +---+--------+----------+-------+---+------------------------+
    | ID|Forename|Middlename|Surname|Sex|FLAG_MiddlenameisNotNull|
    +---+--------+----------+-------+---+------------------------+
    |  1|   Homer|       Jay|Simpson|  M|                    true|
    |  2|   Marge|    Juliet|Simpson|  F|                    true|
    |  3|    Bart|     Jo-Jo|Simpson|  M|                    true|
    |  3|    Bart|     Jo-Jo|Simpson|  M|                    true|
    |  4|    Lisa|     Marie|Simpson|  F|                    true|
    |  5|  Maggie|      null|Simpson|  F|                   false|
    +---+--------+----------+-------+---+------------------------+

    > flag_check(df, prefix ='FLAG_', flags=None,  mode='master', summary=False).show()
    +---+--------+----------+-------+---+------------------------+----------+-----+
    | ID|Forename|Middlename|Surname|Sex|FLAG_MiddlenameisNotNull|flag_count| FAIL|
    +---+--------+----------+-------+---+------------------------+----------+-----+
    |  1|   Homer|       Jay|Simpson|  M|                    true|         1| true|
    |  2|   Marge|    Juliet|Simpson|  F|                    true|         1| true|
    |  3|    Bart|     Jo-Jo|Simpson|  M|                    true|         1| true|
    |  3|    Bart|     Jo-Jo|Simpson|  M|                    true|         1| true|
    |  4|    Lisa|     Marie|Simpson|  F|                    true|         1| true|
    |  5|  Maggie|      null|Simpson|  F|                   false|         0|false|
    +---+--------+----------+-------+---+------------------------+----------+-----+

    See Also
    --------
    flag_summary()

    Notes
    -----
    In all instances summary will be the last component returned e.g. master,summary
    """
    if flags is None:
        try:
            flags = [
                column for column in df.columns if column.startswith(prefix)]
        except:
            print("No flag columns found! Please specify which flag column to summarise\
            with the flags = argument, or specify the correct prefix")

    df = df.withColumn('flag_count', F.lit(0))

    for flag in flags:

        df = (df
              .withColumn('flag_count',
                          F.when(F.col(flag) == True,
                                 F.col('flag_count')+1)
                          .otherwise(F.col('flag_count')))
              )

    df = df.withColumn("FAIL",
                       reduce(add, [F.col(flag).cast(IntegerType())
                                    for flag in flags]))
    df = df.withColumn('FAIL', F.col('FAIL') > 0)

    if summary == True:

        summary_df = flag_summary(df, flags+['FAIL'], pandas=False)

        if mode == 'master':
            return (df,
                    summary_df)

        if mode == 'split':
            return ((df.where(F.col('Fail') == False)),
                    (df.where(F.col('Fail') == True)),
                    summary_df)

        if mode == 'pass':
            return (df.where(F.col('Fail') == False),
                    summary_df)

        if mode == 'fail':
            return (df.where(F.col('Fail') == True),
                    summary_df)

    else:
        if mode == 'master':
            return df

        if mode == 'split':
            return ((df.where(F.col('Fail') == False)),
                    (df.where(F.col('Fail') == True)))

        if mode == 'pass':
            return df.where(F.col('Fail') == False)

        if mode == 'fail':
            return df.where(F.col('Fail') == True)
