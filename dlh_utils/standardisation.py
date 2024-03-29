'''
Functions used to standardise and clean data prior to linkage.
'''
import pyspark.sql.functions as F
from pyspark.sql.types import IntegerType, StringType, StructType, StructField
from pyspark.sql import SparkSession
from dlh_utils import dataframes as da

###############################################################################


def cast_type(df, subset=None, types='string'):
    """
    Casts specific dataframe columns to a specified type.

    The function can either take a subset of columns or if no subset is defined
    it takes all the columns and converts their datatypes into the datatype
    specified by the user.

    Parameters
    ----------
    df : dataframe
      The dataframe to which the function is applied.
    subset: (default = None), {list, string}
      The subset of columns that are having their datatypes converted.
      If this is left blank then it defaults to all columns.
    types: str, (default = string)
      The datatype that the column values will be converted into.

    Returns
    -------
    dataframe
      Returns the complete dataframe with changes to the datatypes on specified
      columns.

    Raises
    -------
    None at present.

    Example
    -------

    > df.show()
    +---+---------+----------------+------------+----------+---+--------+
    | ID|Forename |      Middlename|     Surname|       DoB|Sex|Postcode|
    +---+---------+----------------+------------+----------+---+--------+
    |  1|    David|       Frederick|Attenborough|1926-05-08|  M| KT2 5EQ|
    |  2|  Idrissa|           Akuna|        Elba|1972-09-06|  M|  E1 7AD|
    |  3|   Edward|     Christopher|     Sheeran|1991-02-17|  M| CB7 GGJ|
    |  3|   Edward|     Christopher|     Sheeran|1991-02-17|  M| CB7 GGJ|
    |  4|   Gordon|           James|      Ramsay|1966-11-08|  M|  E1 6AN|
    |  5|     Emma|Charlotte Duerre|      Watson|1990-04-15|  F|EC1A 1AA|
    +---+---------+----------------+------------+----------+---+--------+

    > df.printSchema()
     |-- ID: string (nullable = true)
     |-- Forename: string (nullable = true)
     |-- Middlename: string (nullable = true)
     |-- Surname: string (nullable = true)
     |-- DoB: string (nullable = true)
     |-- Sex: string (nullable = true)
     |-- Postcode: string (nullable = true)

    > df = cast_type(df, subset = ['DOB'], types = 'date')

    > df.printSchema()
     |-- ID: string (nullable = true)
     |-- Forename: string (nullable = true)
     |-- Middlename: string (nullable = true)
     |-- Surname: string (nullable = true)
     |-- DOB: date (nullable = true)
     |-- Sex: string (nullable = true)
     |-- Postcode: string (nullable = true)

    """
    if subset is None:
        subset = df.columns

    if not isinstance(subset, list):
        subset = [subset]

    for col in subset:
        if df.select(col).dtypes[0][1] != types:
            df = df.withColumn(col, F.col(col).cast(types))
    return df

###############################################################################


def standardise_white_space(df, subset=None, wsl='one', fill=None):
    """
    Alters the number of white spaces in the dataframe.

    This can be used to select specified
    columns within the dataframe and alter the number of
    white space characters within the values of those
    specified columns.


    Parameters
    ----------
    df : dataframe
      The dataframe to which the function is applied.
    subset : str or list of str, (default = None)
      The subset of columns that are having their
      white space characters changed. If this is left blank
      then it defaults to all columns.
    wsl : {'one','none'}, (default = 'one')
      wsl stands for white space level, which is used to
      indicate what the user would like white space to be
      replaced with. 'one' replaces multiple whitespces with single
      whitespace. 'none' removes all whitespace
    fill : str, (default = None)
      fill allows user to specify a string that
      would replace whitespace characters.

    Returns
    -------
    dataframe
      The function returns the new dataframe once
      the specified rules are applied.

    Raises
    -------
    None at present.
    """

    if subset is None:
        subset = df.columns

    if not isinstance(subset, list):
        subset = [subset]

    for col in subset:
        if fill is not None:
            if df.select(col).dtypes[0][1] == 'string':
                df = df.withColumn(col, F.trim(F.col(col)))
                df = df.withColumn(col, F.regexp_replace(
                    F.col(col), "\\s+", fill))

        elif wsl == 'none':
            if df.select(col).dtypes[0][1] == 'string':
                df = df.withColumn(col, F.trim(F.col(col)))
                df = df.withColumn(
                    col, F.regexp_replace(F.col(col), "\\s+", ""))
        elif wsl == 'one':
            if df.select(col).dtypes[0][1] == 'string':
                df = df.withColumn(col, F.trim(F.col(col)))
                df = df.withColumn(
                    col, F.regexp_replace(F.col(col), "\\s+", " "))

    return df

###############################################################################


def align_forenames(df, first_name, middle_name, identifier, sep=' '):
    """
    Standardises misalignments of first and middle names by splitting
    the string in specified column(s) on a specified separator and appending
    new columns to the dataframe with each part of the split string.

    Data are then repartitioned on a specified unique identifier column for
    performance purposes.

    Note - if either middle_name or first_name column is not present in data set,
    this needs to be created as a null column before applying this function

    Parameters
    ----------
    df : dataframe
      The dataframe the function is applied to.
    first_name : string
      The title of the column containing first name or all forenames if
      middle name variable is not included
    middle_name : string
      The title of the column containing middle names or previously created
      middle name null column if middle name variable is not included
    identifier: string
      The title of the column containing unique person identifier
    sep : string, default = " "
      The string separating name words in value string

    Returns
    -------
    dataframe
      Dataframe with aligned forename variables (first_name and middle_name)

    Raises
    -------
    None at present.

    See Also
    --------
    dataframes.concat()
    """

    out = df.where(~(F.col(first_name).contains(sep)) | (F.col(first_name).isNull()))
    df = df.where(F.col(first_name).contains(sep))

    df = da.concat(df, 'align_forenames', sep, [
        first_name,
        middle_name
    ])

    df = df.withColumn(first_name, F.regexp_extract(
        F.col('align_forenames'), "([^\s]+)", 1))
    df = df.withColumn(middle_name, F.regexp_extract(
        F.col('align_forenames'), "(\s)(.*)", 2))

    df = df.drop('align_forenames')

    df = out.unionByName(df)

    df = df.repartition(identifier)

    return df

##############################################################################


def reg_replace(df, replace_dict, subset=None):
    """
    Uses regular expressions to replace values within dataframe columns.

    Parameters
    ----------
    df : dataframe
      The dataframe to which the function is applied.
    dic : dictionary
      The values of the dictionary are the substrings
      that are being replaced within the subset columns.
      These need to be regex statements in the form of a
      string. The key is the replacement. The value is the
      regex to be replaced.
    subset : str or list of str, default = None
      The subset is the list of columns in the dataframe
      on which reg_replace is performing its actions.
      If no subset is entered the None default makes sure
      that all columns in the dataframe are in the subset.

    Returns
    -------
    dataframe
      reg_replace returns the dataframe with the column values
      changed appropriately.

    Raises
    -------
    None at present.
    """
    if subset is None:
        subset = df.columns

    if not isinstance(subset, list):
        subset = [subset]

    if subset is not None:
        for col in subset:
            for key, val in replace_dict.items():
                df = df.withColumn(col, F.regexp_replace(F.col(col), val, key))

    return df

##########################################################################


def clean_surname(df, subset):
    '''
    Removes invalid surname response values from a column, as specified
    by components list within function. Concatenates common surname prefixes
    to surname, as specified by prefixes list within function.

    Parameters
    ----------
    df : dataframe
      The dataframe to which the function is applied.
    subset : str or list of str
      The columns on which this function will be applied

    Returns
    -------
    dataframe
      Dataframe with standardised surname variables.

    Raises
    -------
    None at present.

    '''

    components = [
        'NO SURNAME',
        'SURNAME',
    ]

    surname_regex = "|".join([f"\\b{component}\\b"
                              for component in components])

    prefixes = [
        'DE',
        'DA',
        'DU',
        'ST',
        'MC',
        'MAC',
        'VAN',
        'VON',
        'LA',
        'LE',
        'O',
        'AL',
        'DER',
        'EL',
        'DI',
        'DEL',
        'UL',
        'BIN',
        'SAN',
        'BA'
    ]

    surname_prefix_regex = "|".join([f"(?<=\\b{prefix})[ -]"
                                     for prefix in prefixes])

    surname_regex = surname_regex + "|" + surname_prefix_regex

    if not isinstance(subset, list):
        subset = [subset]

    for col in subset:

        df = df.withColumn(col, F.regexp_replace(F.col(col),
                                                 surname_regex,
                                                 ''))

    return df

############################################################################


def clean_forename(df, subset):
    '''
    Removes common forename prefixes from a specified column.

    Parameters
    ----------
    df : dataframe
      The dataframe to which the function is applied.
    subset : str or list of str
      The columns on which this function will be applied.

    Returns
    -------
    dataframe
      Dataframe with standardised forename variables.

    Raises
    -------
    None at present.

    '''

    components = [
        'MR',
        'MRS',
        'DR',
        'MISS',
        'NO NAME',
        'NAME',
        'FORENAME',
        'MS',
        'MX',
        'MSTR',
        'PROF',
        'SIR',
        'LADY'
    ]

    forename_regex = "|".join([f"\\b{component}\\b"
                               for component in components])

    if not isinstance(subset, list):
        subset = [subset]

    for col in subset:

        df = df.withColumn(col, F.regexp_replace(F.col(col),
                                                 forename_regex,
                                                 ''))

    return df

##########################################################################


def group_single_characters(df, subset=None, include_terminals=False):
    """
    Remove spaces between single characters

    The function takes a dataframe and specified columns.
    It then looks for single characters within the column
    values and groups them together by removing the whitespace
    around them.


    Parameters
    ----------
    df : dataframe
      The dataframe to which the function is applied.
    subset : str or list of str, default = None
      Subset of columns on which the function is applied.
      If left as None, the function applies to all
      columns. If the user gives a string value it
      is converted to a list of one column.
      It can also take a list of strings.
    include_terminals : boolean, default = False
      If False, single characters in the first and last
      positions will not be grouped; for example, "a bc d" will
      be returned as "a bc d" without grouping the "a" and "d"
      If True, single characters in the first and last
      positions will be grouped; for example, "a bc d" will
      be grouped as "abcd"

    Returns
    -------
    dataframe
      Dataframe with the same columns but with the
      group single characters function applied to
      the specified columns.

    Raises
    ------
    None at present.
    """

    if subset is None:
        subset = df.columns

    if not isinstance(subset, list):
        subset = [subset]

    regex = r"(?<= \w|^\w|^)[ ]+(?=\w |\w$|$)"
    if include_terminals:
        regex += r"|(?<=^\w)[ ]+(?=\w)"   # Initial single letter
        regex += r"|(?<=\w)[ ]+(?=\w$)"   # Final single letter

    for col in subset:

        df = (df
              .withColumn(col,
                          F.regexp_replace(F.col(col), regex, "")
                          )
              )

    return df

##########################################################################


def clean_hyphens(df, subset=None):
    """
    This cleans the hyphens within the columns
    specified by the function. It does this by removing
    hyphens at the start and end of string values and removes
    the whitespace around the hyphens.

    Parameters
    ----------
    df : dataframe
      Dataframe to which the function is applied.
    subset : str or list of str, default = None
      Subset of columns on which the function is applied.
      If left as None, the function applies to all
      columns. If the user gives a string value it
      is converted to a list of one column.
      It can also take a list of strings.

    Returns
    -------
    dataframe
      Dataframe with the same columns but with the
      clean hyphen function applied to the specified
      columns.

    Raises
    ------
    None at present.
    """

    if subset is None:
        subset = df.columns

    if not isinstance(subset, list):
        subset = [subset]

    for col in subset:

        df = df.withColumn(col, F.regexp_replace(F.col(col),
                                                 "[ ]+[-]+[ ]+|[-]+[ ]+|[ ]+[-]",
                                                 '-'))

        df = df.withColumn(col, F.regexp_replace(F.col(col),
                                                 "^[-]+|[-]+$",
                                                 ''))

    return df

##########################################################################


def standardise_null(df, replace, subset=None, replace_with=None,
                     regex=True):
    """
    Casts values used as nulls to None type (true null).

    If regex is True, this allows user to enter a regular expression that
    captures all values to be treated as null and casts these to specified
    values. Otherwise, the exact values entered in replace are searched
    for and replaced in the data.

    Parameters
    ----------
    df : dataframe
      Dataframe to which the function is applied.
    replace: data-type
      These are the artificial null values
      that are being replaced.
    subset : str or list of str, default = None
      Subset of columns on which the function is applied.
      If left as None, the function applies to all
      columns. If the user gives a string value it
      is converted to a list of one column.
      It can also take a list of strings.
    replace_with : data-type, default = None,
      This is the value the artificial null values
      are converted to (by default None or true null)
    regex: {True,False}, default = True
      The regex argument is a boolean value.
      Setting it to True ensures the replace value
      is treated like a regex statement.
      Setting it to False makes sure the replacement
      value is not treated like a regex statement.

    Returns
    -------
    dataframe
      The function returns the dataframe with
      artificial nulls corrected to the replace value
      in the subset of columns on which the function is applied.

    Raises
    -------
    None at present.

    Example
    -------

    > df.show()
    +---+--------+----------+-------+----------+---+--------+
    | ID|Forename|Middlename|Surname|       DoB|Sex|Postcode|
    +---+--------+----------+-------+----------+---+--------+
    |  1|   Homer|       Jay|     -9|1983-05-12|  M|ET74 2SP|
    |  2|   Marge|        -9|Simpson|1983-03-19|  F|ET74 2SP|
    |  3|      -9|     Jo-Jo|Simpson|2012-04-01| -9|ET74 2SP|
    |  3|    Bart|     Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|
    |  4|    Lisa|     Marie|Simpson|2014-05-09|  F|ET74 2SP|
    |  5|  Maggie|      null|     -9|2021-01-12| -9|ET74 2SP|
    +---+--------+----------+-------+----------+---+--------+

    > standardise_null(df, replace = '-9', subset=None,replace_with='DONUTS',
                       regex=False).show()
    +---+--------+----------+-------+----------+------+--------+
    | ID|Forename|Middlename|Surname|       DoB|   Sex|Postcode|
    +---+--------+----------+-------+----------+------+--------+
    |  1|   Homer|       Jay| DONUTS|1983-05-12|     M|ET74 2SP|
    |  2|   Marge|    DONUTS|Simpson|1983-03-19|     F|ET74 2SP|
    |  3|  DONUTS|     Jo-Jo|Simpson|2012-04-01|DONUTS|ET74 2SP|
    |  3|    Bart|     Jo-Jo|Simpson|2012-04-01|     M|ET74 2SP|
    |  4|    Lisa|     Marie|Simpson|2014-05-09|     F|ET74 2SP|
    |  5|  Maggie|      null| DONUTS|2021-01-12|DONUTS|ET74 2SP|
    +---+--------+----------+-------+----------+------+--------+
    """

    if subset is None:
        subset = df.columns

    if not isinstance(subset, list):
        subset = [subset]

    if regex is True:

        for column in subset:

            df = (df
                  .withColumn(column,
                              F.when(F.col(column).rlike(
                                  replace), replace_with)
                              .otherwise(F.col(column)))
                  )

    if regex is False:

        for column in subset:

            df = (df
                  .withColumn(column,
                              F.when(F.col(column).like(replace), replace_with)
                              .otherwise(F.col(column)))
                  )

    return df


#################################################################


def max_white_space(df, limit, subset=None):
    """
    Sets maximum number of whitespaces in a row.

    Any whitespace above the limit e.g. 3 subsequent whitespaces
    when the limit is 2 will be changed to the limit value within
    the string.

    Parameters
    ----------
    df : dataframe
      Dataframe to which the function is applied.
    limit : int
      This is the maximum value subsequent whitespaces
      are limited to. Any subsequent whitespaces above the
      limit value will be reduced to the limit value.
    subset : str or list of str, default = None
      The subset is the column(s) on which the function is applied.
      If None the function applies to the whole dataframe.

    Returns
    -------
    dataframe
      Function returns dataframe with whitespaces
      limited to the limit value in the chosen subset
      of columns.

    Raises
    -------
    None at present.

    Example
    -------

    > df.show()
    +---+----------+----------+-------------+----------+---+----------------+
    | ID|  Forename|Middlename|      Surname|       DoB|Sex|        Postcode|
    +---+----------+----------+-------------+----------+---+----------------+
    |  1|  Hom   er|       Jay|           -9|1983-05-12|  M|        ET74 2SP|
    |  2|   Ma  rge|        -9|Simpson      |1983-03-19|  F|ET74         2SP|
    |  3|        -9|     Jo-Jo|      Simpson|2012-04-01| -9|      ET74   2SP|
    |  3|      Bart|     Jo-Jo|      Simpson|2012-04-01|  M|        ET74 2SP|
    |  4|      Lisa|     Marie|      Simpson|2014-05-09|  F|        ET74 2SP|
    |  5|Maggie    |      null|           -9|2021-01-12| -9|        ET74 2SP|
    +---+----------+----------+-------------+----------+---+----------------+

    > max_white_space(df, limit = 1, subset=None).show()
    +---+--------+----------+-------+----------+---+--------+
    | ID|Forename|Middlename|Surname|       DoB|Sex|Postcode|
    +---+--------+----------+-------+----------+---+--------+
    |  1|   Homer|       Jay|     -9|1983-05-12|  M|ET74 2SP|
    |  2|   Marge|        -9|Simpson|1983-03-19|  F| ET742SP|
    |  3|      -9|     Jo-Jo|Simpson|2012-04-01| -9| ET742SP|
    |  3|    Bart|     Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|
    |  4|    Lisa|     Marie|Simpson|2014-05-09|  F|ET74 2SP|
    |  5|  Maggie|      null|     -9|2021-01-12| -9|ET74 2SP|
    +---+--------+----------+-------+----------+---+--------+
    """

    if subset is None:
        subset = df.columns

    if not isinstance(subset, list):
        subset = [subset]

    for col in subset:

        df = (df
              .withColumn('space_count',
                          F.length(
                              F.regexp_replace(F.col(col), "[^ ]",
                                               "")))
              .withColumn(col, F.when(F.col('space_count') > limit,
                                      F.regexp_replace(F.col(col), " ", ""))
                          .otherwise(F.col(col)))
              .drop('space_count')
              )

    return df


#################################################################

def max_hyphen(df, limit, subset=None):
    """
    Sets maximum number of hyphens in a row.

    Any hyphens above the limit e.g. 3 subsequent hyphens when the
    limit is 2 will be changed to the limit value within the string.

    Parameters
    ----------
    df : dataframe
      Dataframe to which the function is applied.
    limit : int
      This is the maximum value subsequent hyphens
      are limited to. Any subsequent hyphens above the
      limit value will be reduced to the limit value.
    subset : str or list of str, default = None
      The subset is the column(s) on which the function is applied.
      If None the function applies to the whole dataframe.

    Returns
    -------
    dataframe
      Function returns dataframe with hyphens
      limited to the limit value in the chosen subset
      of columns.

    Raises
    -------
    None at present.

    Example
    -------
    > df.show()
    +---+--------+----------+---------+----------+---+--------+
    | ID|Forename|Middlename|  Surname|       DoB|Sex|Postcode|
    +---+--------+----------+---------+----------+---+--------+
    |  1|   Homer|       Jay|Simp--son|1983-05-12|  M|ET74 2SP|
    |  2|   Marge| Jul---iet|Simp--son|1983-03-19|  F|ET74 2SP|
    |  3|    Bart|  Jo----Jo|Simp--son|2012-04-01|  M|ET74 2SP|
    |  3|    Bart|     Jo-Jo|  Simpson|2012-04-01|  M|ET74 2SP|
    |  4|    Lisa|     Marie|  Simpson|2014-05-09|  F|ET74 2SP|
    |  5|  Maggie|      null|Simp--son|2021-01-12|  F|ET74 2SP|
    +---+--------+----------+---------+----------+---+--------+

    > max_hyphen(df, limit = 1, subset=None).show()
    +---+--------+----------+--------+----------+---+--------+
    | ID|Forename|Middlename| Surname|       DoB|Sex|Postcode|
    +---+--------+----------+--------+----------+---+--------+
    |  1|   Homer|       Jay|Simp-son|1983-05-12|  M|ET74 2SP|
    |  2|   Marge|   Jul-iet|Simp-son|1983-03-19|  F|ET74 2SP|
    |  3|    Bart|     Jo-Jo|Simp-son|2012-04-01|  M|ET74 2SP|
    |  3|    Bart|     Jo-Jo| Simpson|2012-04-01|  M|ET74 2SP|
    |  4|    Lisa|     Marie| Simpson|2014-05-09|  F|ET74 2SP|
    |  5|  Maggie|      null|Simp-son|2021-01-12|  F|ET74 2SP|
    +---+--------+----------+--------+----------+---+--------+

    """

    if subset is None:
        subset = df.columns

    if not isinstance(subset, list):
        subset = [subset]

    number_hyphens = "-" * limit

    for col in subset:

        df = (df
              .withColumn('space_count',
                          F.length(
                              F.regexp_replace(F.col(col), "[^-]",
                                               "")))
              .withColumn(col, F.when(F.col('space_count') > limit,
                                      F.regexp_replace(F.col(col), r"(-)\1{1,}", number_hyphens))
                          .otherwise(F.col(col)))
              .drop('space_count')
              )

    return df
#################################################################


def remove_punct(df, subset=None, keep=None):
    """
    Removes punctuation from strings.

    Where the keep value is set, it will remove all punctuation
    from given columns other than the punctuation value set by
    the keep argument.

    Parameters
    ----------
    df : dataframe
      Dataframe to which the function is applied.
    subset :  str or list of str, default = None
      The subset is the column(s) on which the function is applied.
      If None the function applies to the whole dataframe.
    keep: str or list of str, default = None
      Can be set to a string with a symbol you want to keep, e.g.
      if you want to keep the addition symbol set keep = '+' and
      it will refrain from removing the + symbol from the
      subset of columns.

    Returns
    -------
    dataframe
      Dataframe with the same columns but with the
      remove_punct function applied to the specified
      columns.

    Raises
    -------
    None at present.

    Example
    -------
    > df.show()
    +---+--------+----------+---------+----------+---+--------+
    | ID|Forename|Middlename|  Surname|       DoB|Sex|Postcode|
    +---+--------+----------+---------+----------+---+--------+
    |  1|  Ho'mer|       Jay|Simp--son|1983-05-12|  M|ET74 2SP|
    |  2|   Marge| Jul---iet|Simp--son|1983-03-19|  F|ET74 2SP|
    |  3|    Bart|  Jo----Jo|Simp--son|2012-04-01|  M|ET74 2SP|
    |  3|    Bart|     Jo-Jo|  Simpson|2012-04-01|  M|ET74 2SP|
    |  4|    Lisa|     Marie|  Simpson|2014-05-09|  F|ET74 2SP|
    |  5| Ma'ggie|      null|Simp--son|2021-01-12|  F|ET74 2SP|
    +---+--------+----------+---------+----------+---+--------+

    > remove_punct(df,subset=None,keep=None).show()
    +---+--------+----------+-------+--------+---+--------+
    | ID|Forename|Middlename|Surname|     DoB|Sex|Postcode|
    +---+--------+----------+-------+--------+---+--------+
    |  1|   Homer|       Jay|Simpson|19830512|  M|ET74 2SP|
    |  2|   Marge|    Juliet|Simpson|19830319|  F|ET74 2SP|
    |  3|    Bart|      JoJo|Simpson|20120401|  M|ET74 2SP|
    |  3|    Bart|      JoJo|Simpson|20120401|  M|ET74 2SP|
    |  4|    Lisa|     Marie|Simpson|20140509|  F|ET74 2SP|
    |  5|  Maggie|      null|Simpson|20210112|  F|ET74 2SP|
    +---+--------+----------+-------+--------+---+--------+

    """

    if subset is None:
        subset = df.columns

    if not isinstance(subset, list):
        subset = [subset]

    if keep is not None:
        keep = "".join(keep)
        regex = f"[^A-Za-z0-9 {keep}]"
    else:
        regex = "[^A-Za-z0-9 ]"

    for x in subset:
        df = df.withColumn(x, F.regexp_replace(F.col(x), regex, ""))
    return df

#####################################################################


def standardise_case(df, subset=None, val='upper'):
    """
    Converts the case of all specified variables in a dataframe into
    one format. For example, using upper will set the case of all
    specified variables to UPPER CASE.


    Parameters
    ----------
    df : dataframe
      Dataframe to which the function is applied.
    subset :  str or list of str, default = None
      The subset is the column(s) on which the function is applied.
      If None the function applies to the whole dataframe.
    val: {'upper','lower','title'}, default = 'upper'
      Takes three types of string values
      and changes the values in the subset columns to the
      case type respectively.

    Returns
    -------
    dataframe
      Function returns dataframe with case conversions applied.

    Raises
    -------
      None at present.

    Example
    -------
    > df.show()
    +---+--------+----------+-------+----------+---+--------+
    | ID|Forename|Middlename|Surname|       DoB|Sex|Postcode|
    +---+--------+----------+-------+----------+---+--------+
    |  1|   Homer|       Jay|SiMpSoN|1983-05-12|  M|ET74 2SP|
    |  2|   mARGE|    juliet|SiMpSoN|1983-03-19|  F|ET74 2SP|
    |  3|    Bart|      JOJo|SiMpSoN|2012-04-01|  M|ET74 2SP|
    |  3|    Bart|     Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|
    |  4|    LISA|     Marie|SiMpSoN|2014-05-09|  F|ET74 2SP|
    |  5|  MAGGie|      null|SiMpSoN|2021-01-12|  F|ET74 2SP|
    +---+--------+----------+-------+----------+---+--------+

    > standardise_case(df,subset = None, val='upper').show()
    +---+--------+----------+-------+----------+---+--------+
    | ID|Forename|Middlename|Surname|       DoB|Sex|Postcode|
    +---+--------+----------+-------+----------+---+--------+
    |  1|   HOMER|       JAY|SIMPSON|1983-05-12|  M|ET74 2SP|
    |  2|   MARGE|    JULIET|SIMPSON|1983-03-19|  F|ET74 2SP|
    |  3|    BART|      JOJO|SIMPSON|2012-04-01|  M|ET74 2SP|
    |  3|    BART|     JO-JO|SIMPSON|2012-04-01|  M|ET74 2SP|
    |  4|    LISA|     MARIE|SIMPSON|2014-05-09|  F|ET74 2SP|
    |  5|  MAGGIE|      null|SIMPSON|2021-01-12|  F|ET74 2SP|
    +---+--------+----------+-------+----------+---+--------+

    """

    if subset is None:
        subset = df.columns

    if not isinstance(subset, list):
        subset = [subset]

    for col in subset:
        if val == 'upper':
            if df.select(col).dtypes[0][1] == 'string':
                df = df.withColumn(col, F.upper(F.col(col)))
        if val == 'lower':
            if df.select(col).dtypes[0][1] == 'string':
                df = df.withColumn(col, F.lower(F.col(col)))
        if val == 'title':
            if df.select(col).dtypes[0][1] == 'string':
                df = df.withColumn(col, F.initcap(F.col(col)))
    return df

##########################################################################


def trim(df, subset=None):
    """
    Removes leading and trailing whitespace from all
    or selected string columns.

    Parameters
    ----------
    df : dataframe
      Dataframe to which the function is applied..
    subset : default = None, string or list of strings
      The subset is the column(s) on which the function is applied.
      If None the function applies to the whole dataframe.

    Returns
    -------
    dataframe
      Dataframe with leading and trailing whitespace removed
      from selected or all string columns in place.

    Raises
    -------
    None at present.

    Example
    -------
    > df.show()

    +---+---------+----------+-------------+----------+---+--------+
    | ID| Forename|Middlename|      Surname|       DoB|Sex|Postcode|
    +---+---------+----------+-------------+----------+---+--------+
    |  1|    Homer|       Jay|      SiMpSoN|1983-05-12|  M|ET74 2SP|
    |  2|    mARGE|    juliet|   SiMpSoN   |1983-03-19|  F|ET74 2SP|
    |  3|     Bart|      JOJo|   SiMpSoN   |2012-04-01|  M|ET74 2SP|
    |  3|     Bart|     Jo-Jo|      Simpson|2012-04-01|  M|ET74 2SP|
    |  4|     LISA|    Marie |   SiMpSoN   |2014-05-09|  F|ET74 2SP|
    |  5|MAGGie   |      null|      SiMpSoN|2021-01-12|  F|ET74 2SP|
    +---+---------+----------+-------------+----------+---+--------+

    > trim(df,subset=None).show()

    +---+--------+----------+-------+----------+---+--------+
    | ID|Forename|Middlename|Surname|       DoB|Sex|Postcode|
    +---+--------+----------+-------+----------+---+--------+
    |  1|   Homer|       Jay|SiMpSoN|1983-05-12|  M|ET74 2SP|
    |  2|   mARGE|    juliet|SiMpSoN|1983-03-19|  F|ET74 2SP|
    |  3|    Bart|      JOJo|SiMpSoN|2012-04-01|  M|ET74 2SP|
    |  3|    Bart|     Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|
    |  4|    LISA|     Marie|SiMpSoN|2014-05-09|  F|ET74 2SP|
    |  5|  MAGGie|      null|SiMpSoN|2021-01-12|  F|ET74 2SP|
    +---+--------+----------+-------+----------+---+--------+
    """

    if subset is None:
        subset = df.columns

    if not isinstance(subset, list):
        subset = [subset]

    types = [x for x in df.dtypes
             if x[0] in subset]

    types = dict(zip([x[0] for x in types],
                     [x[1] for x in types]))

    for col in subset:

        if types[col] == 'string':

            df = df.withColumn(col, F.trim(F.col(col)))

    return df

##########################################################################


def add_leading_zeros(df, subset, n):
    """
    Adds leading zeros to the numeric characters of a string, until the
    length of the string equals n. For example if a string is 1 and n
    is set to be 7, then the result would be 0000001.

    Parameters
    ----------
    df : dataframe
      Dataframe to which the function is applied.
    subset : string or list of strings, default = None
      The subset is the column(s) on which the function is applied.
      If None the function applies to the whole dataframe.
    n: int
      This is the length to which the string is adjusted.

    Returns
    -------
    dataframe
      Returns the dataframe with the subset columns
      padded with 0s dependant on the value of n.

    Raises
    -------
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

    > add_leading_zeros(df,subset = 'ID',n = 3).show()

    +---+--------+----------+-------+----------+---+--------+
    | ID|Forename|Middlename|Surname|       DoB|Sex|Postcode|
    +---+--------+----------+-------+----------+---+--------+
    |001|   Homer|       Jay|Simpson|1983-05-12|  M|ET74 2SP|
    |002|   Marge|    Juliet|Simpson|1983-03-19|  F|ET74 2SP|
    |003|    Bart|     Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|
    |003|    Bart|     Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|
    |004|    Lisa|     Marie|Simpson|2014-05-09|  F|ET74 2SP|
    |005|  Maggie|      null|Simpson|2021-01-12|  F|ET74 2SP|
    +---+--------+----------+-------+----------+---+--------+
    """

    if not isinstance(subset, list):
        subset = [subset]

    for col in subset:

        df = (df
              .withColumn(col,
                          F.lpad(F.col(col), n, '0')
                          )
              )

    return df

##############################################################################


def replace(df, subset, replace_dict, use_join=False, use_regex=False):
    """
    Replaces specific string values in given column(s) with specified values.

    Parameters
    ----------
    df : dataframe
      Dataframe to which the function is applied.
    subset : string or list of strings
      The subset is the column(s) on which the function is applied.
      If None the function applies to the whole dataframe.
    replace_dict: dictionary
      Dictionary given needs to be in the format of
      value_to_be_replaced:value_to_replace_with.
    use_join : Boolean, default = False
      Set this to True for higher performance if you have a large
      number of key-value pairs (e.g. industrial classifications).
      NOTE: You cannot use this mode to replace None values.
      Call replace() separately with use_join=False if you need to
      replace the None values.
    use_regex : Boolean, default = False
      Use regular expression matching. By default replacements are
      only done if the whole value matches the dictionary key exactly.
      If use_regex is True, use_join must be set to False.

    Returns
    -------
    dataframe
      Dataframe with replaced values as specified.

    Raises
    -------
    ValueError if use_join and use_regex are both passed in as True.

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

    > replace(df,subset = 'Forename',replace_dict = {'Bart': 'Turbo man'}).show()

    +---+---------+----------+-------+----------+---+--------+
    | ID| Forename|Middlename|Surname|       DoB|Sex|Postcode|
    +---+---------+----------+-------+----------+---+--------+
    |  1|    Homer|       Jay|Simpson|1983-05-12|  M|ET74 2SP|
    |  2|    Marge|    Juliet|Simpson|1983-03-19|  F|ET74 2SP|
    |  3|Turbo man|     Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|
    |  3|Turbo man|     Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|
    |  4|     Lisa|     Marie|Simpson|2014-05-09|  F|ET74 2SP|
    |  5|   Maggie|      null|Simpson|2021-01-12|  F|ET74 2SP|
    +---+---------+----------+-------+----------+---+--------+
    """

    if use_join and use_regex:
        raise ValueError("Can't call replace() with True values for both use_join and use_regex.")
    if use_join:
        for k, i in replace_dict.items():
            if k is None or i is None:
                raise ValueError("Join mode (use_join=True) is not compatible with the use of None\
                                 in the replace dictionary. Set use_join=True to use None values.")

    if not isinstance(subset, list):
        subset = [subset]

    if use_join:
        spark = SparkSession.builder.getOrCreate()
        schema = StructType([
            StructField("_replace_dict_element_before", StringType(), True),
            StructField("_replace_dict_element_after", StringType(), True)])
        replace_df = spark.createDataFrame(replace_dict.items(), schema)
        for col in subset:
            df = df.join(
                replace_df,
                df[col] == replace_df["_replace_dict_element_before"],
                how="left"
            )
            df = df.withColumn(
                col,
                F.coalesce(df["_replace_dict_element_after"], df[col])
            )
            df = df.drop("_replace_dict_element_before", "_replace_dict_element_after")
    else:
        for col in subset:
            for before, after in replace_dict.items():
                if use_regex:
                    df = (df
                          .withColumn(col, F.when(F.col(col).rlike(before), after)
                                      .otherwise(F.col(col)))
                          )
                else:
                    df = (df
                          .withColumn(col, F.when(F.col(col).like(before), after)
                                      .otherwise(F.col(col)))
                          )

    return df

##############################################################################


def standardise_date(df, col_name, in_date_format='dd-MM-yyyy',
                     out_date_format='yyyy-MM-dd',null_counts=False):
    """
    Changes the date format of a specified date column.

    Also has an optional argument null_counts. If set to False, the function
    only returns the dataframe with specified date values altered.
    However, if set to True, will also return a tuple, displaying a count of
    nulls in the dataframe before and after the function has been applied.
    This is because the standardise_date function will make a date
    value null if its format differs from the in_date_format.

    Parameters
    ----------
    df : dataframe
      Dataframe to which the function is applied.
    col_name: string
      The column in the dataframe to which the function is being
      applied.
    in_date_format: default = 'dd-MM-yyyy', string
      This is the current date format of the the column.
    out_date_format: default = 'yyyy-MM-dd', string
      This is the date format to which the column values will be
      changed.
    null_counts: default = False, bool
      If set to True provides a tuple, where the first part displays
      the df null count before the function has been applied, and the
      second part is the df null count after the function has been
      applied.

    Returns
    -------
    If null_counts = False:
    dataframe
      Dataframe with specified date column values altered
      to new specified format.

    If null_counts = True:
    dataframe as described above,
    Tuple
      showing null counts before and after standardise_date
      has been applied to df

    Raises
    -------
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

    > standardise_date(df,col_name = 'DoB',in_date_format = 'yyyy-MM-dd',
                       out_date_format = 'dd/MM/yyyy').show()

    +---+--------+----------+-------+----------+---+--------+
    | ID|Forename|Middlename|Surname|       DoB|Sex|Postcode|
    +---+--------+----------+-------+----------+---+--------+
    |  1|   Homer|       Jay|Simpson|12/05/1983|  M|ET74 2SP|
    |  2|   Marge|    Juliet|Simpson|19/03/1983|  F|ET74 2SP|
    |  3|    Bart|     Jo-Jo|Simpson|01/04/2012|  M|ET74 2SP|
    |  3|    Bart|     Jo-Jo|Simpson|01/04/2012|  M|ET74 2SP|
    |  4|    Lisa|     Marie|Simpson|09/05/2014|  F|ET74 2SP|
    |  5|  Maggie|      null|Simpson|12/01/2021|  F|ET74 2SP|
    +---+--------+----------+-------+----------+---+--------+

    Example using null_counts:
    > df.show()

    +----------+
    |      date|
    +----------+
    |1990/07/19|
    |2020/09/01|
    |1998/03/21|
    |1999/11/02|
    |2005-12-24|
    +----------+

    >df,nulls = standardise_date(df,'date','yyyy/MM/dd','yyyy.MM.dd',True)

    >df.show()
    +----------+
    |      date|
    +----------+
    |1990.07.19|
    |2020.09.01|
    |1998.03.21|
    |1999.11.02|
    |      null|
    +----------+

    >nulls
    (0, 1)

    """
    if null_counts:
        null_before = df.where(F.col(col_name).isNull()).count()

    df = df.withColumn(col_name, F.unix_timestamp(F.col(col_name), in_date_format))
    df = df.withColumn(col_name, F.from_unixtime(F.col(col_name), out_date_format))

    if null_counts:
        null_after = df.where(F.col(col_name).isNull()).count()

        return df,(null_before,null_after)

    else:
        return df

##############################################################################


def fill_nulls(df, fill, subset=None):
    """
    Fills null and NaN with specified value

    Parameters
    ----------
    df : dataframe
      Dataframe to which the function is applied.
    fill: None or a string
      This is the value the NaN/Null values are replaced by.
      The data type of the fill value must match
      that of the column it is replacing values within.
    subset : default = None, string or list of strings
      The subset is the column(s) on which the function is applied.
      If None the function applies to the whole dataframe.


    Returns
    -------
    dataframe
      Dataframe with null/NaN values replaced from
      the subset of columns.

    Raises
    -------
    None at present.

    Example
    -------
    > df.show()

    +---+--------+----------+-------+----------+---+--------+
    | ID|Forename|Middlename|Surname|       DoB|Sex|Postcode|
    +---+--------+----------+-------+----------+---+--------+
    |  1|   Homer|      null|Simpson|1983-05-12|  M|ET74 2SP|
    |  2|   Marge|      null|Simpson|1983-03-19|  F|ET74 2SP|
    |  3|    Bart|      null|Simpson|2012-04-01|  M|ET74 2SP|
    |  3|    Bart|     Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|
    |  4|    Lisa|     Marie|Simpson|2014-05-09|  F|ET74 2SP|
    |  5|  Maggie|      null|Simpson|2021-01-12|  F|ET74 2SP|
    +---+--------+----------+-------+----------+---+--------+

    > fill_nulls(df, fill = 'donuts',subset=None).show()

    +---+--------+----------+-------+----------+---+--------+
    | ID|Forename|Middlename|Surname|       DoB|Sex|Postcode|
    +---+--------+----------+-------+----------+---+--------+
    |  1|   Homer|    donuts|Simpson|1983-05-12|  M|ET74 2SP|
    |  2|   Marge|    donuts|Simpson|1983-03-19|  F|ET74 2SP|
    |  3|    Bart|    donuts|Simpson|2012-04-01|  M|ET74 2SP|
    |  3|    Bart|     Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|
    |  4|    Lisa|     Marie|Simpson|2014-05-09|  F|ET74 2SP|
    |  5|  Maggie|    donuts|Simpson|2021-01-12|  F|ET74 2SP|
    +---+--------+----------+-------+----------+---+--------+
    """

    if subset is None:
        subset = df.columns

    if not isinstance(subset, list):
        subset = [subset]

    for col in subset:
        df = (df
              .withColumn(col, F.when(
                  (F.col(col).isNull()) | (F.isnan(F.col(col))), fill)
                  .otherwise(F.col(col)))
              )

    return df

################################################################################


def age_at(df, reference_col, in_date_format='dd-MM-yyyy', *age_at_dates):
    """
    Calculates individuals' ages at specified dates from a reference Date of Birth column

    The function takes a list of dates, and for each date creates a new column
    specifying an individual's age at that date.

    Parameters
    ----------
    df : dataframe
      Dataframe to which the function is applied.
    reference_column: string
      The original date of birth column needed to calculate age.
    in_date_format: default = 'dd-MM-yyyy',string
      The date format of the date of birth column.
      It uses hyphens or forward slashes to split the date
      up and dd,MM,yyyy to show date month and year respectively.
      e.g. 'dd-MM-yyyy' , 'dd/MM/yyyy', 'yyyy-MM-dd'.
    *age_at_dates: list of strings
      The list of dates at which the user wants to calculate ages. Any
      number of dates can be given. The dates need to be in the
      following format: 'yyyy-MM-dd'.

    Returns
    -------
    dataframe
      Returns the complete dataframe with additional column(s)
      giving age at specified date(s).

    Raises
    -------
    None at present.

    Example
    -------
    > df.show()
    +---+--------+-------+----------+---+
    | ID|Forename|Surname|       DoB|Sex|
    +---+--------+-------+----------+---+
    |  1|   Homer|Simpson|1983-05-12|  M|
    |  2|   Marge|Simpson|1983-03-19|  F|
    |  3|    Bart|Simpson|2012-04-01|  M|
    |  3|    Bart|Simpson|2012-04-01|  M|
    |  4|    Lisa|Simpson|2014-05-09|  F|
    |  5|  Maggie|Simpson|2021-01-12|  F|
    +---+--------+-------+----------+---+

    > dates = ['2022-11-03','2020-12-25']
    > age_at(df,'DoB','yyyy-MM-dd',*dates).show()
    +---+--------+-------+----------+---+---------------------+---------------------+
    | ID|Forename|Surname|       DoB|Sex|DoB_age_at_2022-11-03|DoB_age_at_2020-12-25|
    +---+--------+-------+----------+---+---------------------+---------------------+
    |  1|   Homer|Simpson|1983-05-12|  M|                   39|                   37|
    |  2|   Marge|Simpson|1983-03-19|  F|                   39|                   37|
    |  3|    Bart|Simpson|2012-04-01|  M|                   10|                    8|
    |  3|    Bart|Simpson|2012-04-01|  M|                   10|                    8|
    |  4|    Lisa|Simpson|2014-05-09|  F|                    8|                    6|
    |  5|  Maggie|Simpson|2021-01-12|  F|                    1|                    0|
    +---+--------+-------+----------+---+---------------------+---------------------+

    """

    df = df.withColumn(
        f"{reference_col}_fmt", F.unix_timestamp(F.col(reference_col), in_date_format)
    ).withColumn(
        f"{reference_col}_fmt",
        F.from_unixtime(F.col(f"{reference_col}_fmt"), "yyyy-MM-dd"),
    )

    for age_at_date in age_at_dates:

        df = df.withColumn(
            f"{reference_col}_age_at_{age_at_date}",
            (
                F.months_between(
                    F.lit(age_at_date),
                    F.col(f"{reference_col}_fmt"),
                ) / F.lit(12)
            ).cast(IntegerType()),
        )

    df = df.drop(f"{reference_col}_fmt")

    return df
