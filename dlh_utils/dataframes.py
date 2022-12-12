import re
import pyspark.sql.functions as F
from pyspark.sql import Window
from dlh_utils import standardisation as st

##################################################################################


def select(df, columns=None, startswith=None, endswith=None, contains=None,
           regex=None, drop_duplicates=True):
    """
    Retains only specified list of columns or columns meeting startswith,
    endswith, contains or regex arguments.

    Parameters
    ----------
    df : dataframe
      Dataframe to which the function is applied.

    columns : string or list of strings, default = None
      This argument can be entered as a list of column headers
      that are the columns to be selected. If a single string
      that is a name of a column is entered, it will select
      only that column.

    startswith : string, default = None
      This parameter takes a string value and selects
      columns from the dataframe if the column
      title starts with the string value.

    endswith : string, default = None
      This parameter takes a string value and selects
      columns from the dataframe if the column
      title ends with the string value.

    contains : string, default = None
      This parameter takes a string value and selects
      columns from the dataframe if the column
      title contains the string value.

    regex : string, default = None
      This parameter takes a string value in 
      regex format and selects columns from the 
      dataframe if the column title matches
      the conditions of the regex string.

    drop_duplicates : bool, default = True
      This parameter drops duplicated columns.

    Returns
    -------
    dataframe
      Dataframe with columns limited to those
      specified by the parameters.

    Raises
    ------
    None at present.

    Example
    -------

    data = [("1","6","1","Simpson","1983-05-12","M","ET74 2SP"),
            ("2","8","2","Simpson","1983-03-19","F","ET74 2SP"),
            ("3","7","3","Simpson","2012-04-01","M","ET74 2SP"),
            ("3","9","3","Simpson","2012-04-01","M","ET74 2SP"),
            ("4","9","4","Simpson","2014-05-09","F","ET74 2SP"),
            ("5","6",4,"Simpson","2021-01-12","F","ET74 2SP")]
    df = spark.createDataFrame(data=data,schema=["ID","ID2","clust","ROWNUM","DoB","Sex","Postcode"])

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

    > select(df,columns = None, startswith = 'F').show()
      +--------+
      |Forename|
      +--------+
      |   Homer|
      |   Marge|
      |  Maggie|
      |    Bart|
      |    Lisa|
      +--------+

    > select(df,columns = None, endswith = 'e',drop_duplicates = False).show()
     +--------+----------+-------+--------+
     |Forename|Middlename|Surname|Postcode|
     +--------+----------+-------+--------+
     |   Homer|       Jay|Simpson|ET74 2SP|
     |   Marge|    Juliet|Simpson|ET74 2SP|
     |    Bart|     Jo-Jo|Simpson|ET74 2SP|
     |    Bart|     Jo-Jo|Simpson|ET74 2SP|
     |    Lisa|     Marie|Simpson|ET74 2SP|
     |  Maggie|      null|Simpson|ET74 2SP|
     +--------+----------+-------+--------+

    > select(df,columns = None, contains = 'name').show()
     +--------+----------+-------+
     |Forename|Middlename|Surname|
     +--------+----------+-------+
     |    Bart|     Jo-Jo|Simpson|
     |   Marge|    Juliet|Simpson|
     |   Homer|       Jay|Simpson|
     |    Lisa|     Marie|Simpson|
     |  Maggie|      null|Simpson|
     +--------+----------+-------+

     > select(df,columns = None, regex = '^[A-Z]{2}$').show()
     +---+
     | ID|
     +---+
     |  3|
     |  5|
     |  1|
     |  4|
     |  2|
     +---+

    """
    if columns is not None:
        df = df.select(columns)

    if startswith is not None:
        df = df.select(
            [x for x in df.columns if x.startswith(startswith)]
        )

    if endswith is not None:
        df = df.select(
            [x for x in df.columns if x.endswith(endswith)]
        )

    if contains is not None:
        df = df.select(
            [x for x in df.columns if contains in x]
        )

    if regex is not None:
        df = df.select(
            [x for x in df.columns if re.search(regex, x)]
        )

    if drop_duplicates:
        df = df.dropDuplicates()

    return df

###############################################################################


def drop_columns(df, subset=None, startswith=None, endswith=None, contains=None,
                regex=None, drop_duplicates=True):
    """
    drop_columns allows user to specify one or more columns 
    to be dropped from the dataframe.

    Parameters
    ----------
    df : dataframe
      Dataframe to which the function is applied.

    subset : string or list of strings, default = None
      The subset can be entered as a list of column headers
      that are the columns to be dropped. If a single string
      that is a name of a column is entered, it will drop
      that column.

    startswith : string, default = None
      This parameter takes a string value and drops
      columns from the dataframe if the column
      title starts with the string value.

    endswith : string, default = None
      This parameter takes a string value and drops
      columns from the dataframe if the column
      title ends with the string value.

    contains : string, default = None
      This parameter takes a string value and drops 
      columns from the dataframe if the column
      title contains the string value.

    regex : string, default = None
      This parameter takes a string value in 
      regex format and drops columns from the 
      dataframe if the column title matches
      the conditions of the regex string.

    drop_duplicates : bool, default = True
      This parameter drops duplicated columns.

    Returns
    -------
    dataframe
      Dataframe with columns dropped based on parameters.

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

    > drop_columns(df,subset = None, startswith = 'S').show()
      +---+--------+----------+----------+--------+
      | ID|Forename|Middlename|       DoB|Postcode|
      +---+--------+----------+----------+--------+
      |  2|   Marge|    Juliet|1983-03-19|ET74 2SP|
      |  3|    Bart|     Jo-Jo|2012-04-01|ET74 2SP|
      |  4|    Lisa|     Marie|2014-05-09|ET74 2SP|
      |  5|  Maggie|      null|2021-01-12|ET74 2SP|
      |  1|   Homer|       Jay|1983-05-12|ET74 2SP|
      +---+--------+----------+----------+--------+

    > drop_columns(df,subset = None, endswith = 'e',drop_duplicates = False).show()
     +---+----------+---+
     | ID|       DoB|Sex|
     +---+----------+---+
     |  1|1983-05-12|  M|
     |  2|1983-03-19|  F|
     |  3|2012-04-01|  M|
     |  3|2012-04-01|  M|
     |  4|2014-05-09|  F|
     |  5|2021-01-12|  F|
     +---+----------+---+

    > drop_columns(df,subset = None, contains = 'name').show()
     +---+----------+---+--------+
     | ID|       DoB|Sex|Postcode|
     +---+----------+---+--------+
     |  4|2014-05-09|  F|ET74 2SP|
     |  2|1983-03-19|  F|ET74 2SP|
     |  3|2012-04-01|  M|ET74 2SP|
     |  5|2021-01-12|  F|ET74 2SP|
     |  1|1983-05-12|  M|ET74 2SP|
     +---+----------+---+--------+

     > drop_columns(df,subset = None, regex = '^[A-Z]{2}$').show()
     +--------+----------+-------+----------+---+--------+
     |Forename|Middlename|Surname|       DoB|Sex|Postcode|
     +--------+----------+-------+----------+---+--------+
     |   Marge|    Juliet|Simpson|1983-03-19|  F|ET74 2SP|
     |   Homer|       Jay|Simpson|1983-05-12|  M|ET74 2SP|
     |    Lisa|     Marie|Simpson|2014-05-09|  F|ET74 2SP|
     |    Bart|     Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|
     |  Maggie|      null|Simpson|2021-01-12|  F|ET74 2SP|
     +--------+----------+-------+----------+---+--------+

    """

    if startswith is not None:
        df = df.drop(*
                     [x for x in df.columns if x.startswith(startswith)]
                     )

    if endswith is not None:
        df = df.drop(*
                     [x for x in df.columns if x.endswith(endswith)]
                     )

    if contains is not None:
        df = df.drop(*
                     [x for x in df.columns if contains in x]
                     )

    if regex is not None:
        df = df.drop(*
                     [x for x in df.columns if re.search(regex, x)]
                     )

    if subset != None:
        if type(subset) != list:
            subset = [subset]
        df = df.drop(*subset)

    if drop_duplicates:
        df = df.dropDuplicates()

    return df

###############################################################################


def concat(df, out_col, sep=' ', columns=[]):
    """
    Concatenates strings from specified columns into a single string and stores
    the new string value in a new column.

    Parameters
    ----------
    df : dataframe
      Dataframe to which the function is applied.
    out_col : string
      The name, in string format, of the 
      output column for the new concatenated
      strings to be stored in.
    sep : string, default = ' '
      This is the value used to seperate the
      strings in the different columns when
      combinging them into a single string.
    columns : list, default = []
      The list of columns being concatenated into 
      one string

    Returns
    -------
    dataframe
      Returns dataframe with 'out_col' column
      containing the concatenated string.

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

    > concat(df, out_col = 'Full Name', sep = ' ', columns = ['Forename','Middlename','Surname']).show()
    +---+--------+----------+-------+----------+---+--------+--------------------+
    | ID|Forename|Middlename|Surname|       DoB|Sex|Postcode|           Full Name|
    +---+--------+----------+-------+----------+---+--------+--------------------+
    |  1|   Homer|       Jay|Simpson|1983-05-12|  M|ET74 2SP|   Homer Jay Simpson|
    |  2|   Marge|    Juliet|Simpson|1983-03-19|  F|ET74 2SP|Marge Juliet Simpson|
    |  3|    Bart|     Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|  Bart Jo-Jo Simpson|
    |  3|    Bart|     Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|  Bart Jo-Jo Simpson|
    |  4|    Lisa|     Marie|Simpson|2014-05-09|  F|ET74 2SP|  Lisa Marie Simpson|
    |  5|  Maggie|      null|Simpson|2021-01-12|  F|ET74 2SP|      Maggie Simpson|
    +---+--------+----------+-------+----------+---+--------+--------------------+

    """

    df = (df
          .withColumn(out_col,
                      F.concat_ws(sep, *[F.col(x) for x in columns]))
          )

    if sep != '':

        df = (df
              .withColumn(out_col, F.regexp_replace(F.col(out_col),
                                                    f"[{sep}]+", sep))
              .withColumn(out_col, F.regexp_replace(F.col(out_col),
                                                    f"^[{sep}]|[{sep}]$", ''))
              .withColumn(out_col, F.when(F.col(out_col).rlike("^$"), None)
                          .otherwise(F.col(out_col)))
              )

    return df

#############################################################################


def explode(df, column, on=' ', retain=False, drop_duplicates=True, flag=None):
    """
    Splits a string column on specified separator (default=" ") 
    and creates a new row for each element of the split string array
    maintaining values in all other columns. 

    Parameters
    ----------
    df: dataframe function is being applied to
    column : string 
      column to be exploded 
    on : string, default = ' ' 
      This argument takes a string or regex value
      in string format and explodes the values where 
      either the string or regex value matches.
    retain : bool, default = False
      option to retain original string values.
    drop_duplicates: bool, default = True
      option to drop duplicate values
    flag: string, default = None
      name of flag column, that contains
      False values for rows that are equal. 
      For a flag column to be appended, 
      retain needs to be True. 

    Returns
    -------
    dataframe
      dataframe with additional rows accomodating all elements 
      of exploded string column.

    Raises
    ------
    None at present

    Example
    -------

    > df.show(truncate= False)
    +---+--------+----------+-------+----------+---+--------+----------------------+
    |ID |Forename|Middlename|Surname|DoB       |Sex|Postcode|Description           |
    +---+--------+----------+-------+----------+---+--------+----------------------+
    |1  |Homer   |Jay       |Simpson|1983-05-12|M  |ET74 2SP|Balding Lazy          |
    |2  |Marge   |Juliet    |Simpson|1983-03-19|F  |ET74 2SP|Blue-hair Kind-hearted|
    |3  |Bart    |Jo-Jo     |Simpson|2012-04-01|M  |ET74 2SP|Spikey-hair Rebellious|
    |3  |Bart    |Jo-Jo     |Simpson|2012-04-01|M  |ET74 2SP|Spikey-hair Rebellious|
    |4  |Lisa    |Marie     |Simpson|2014-05-09|F  |ET74 2SP|Red-dress Smart       |
    |5  |Maggie  |null      |Simpson|2021-01-12|F  |ET74 2SP|Star-hair Mute        |
    +---+--------+----------+-------+----------+---+--------+----------------------+

    e.g, if you wanted to separate the record's appearance from their personality description:
    > explode(df,column = 'Description',on = ' ',retain = False,drop_duplicates = True, flag = None).show()
    +---+--------+----------+-------+----------+---+--------+------------+
    | ID|Forename|Middlename|Surname|       DoB|Sex|Postcode| Description|
    +---+--------+----------+-------+----------+---+--------+------------+
    |  3|    Bart|     Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP| Spikey-hair|
    |  1|   Homer|       Jay|Simpson|1983-05-12|  M|ET74 2SP|     Balding|
    |  4|    Lisa|     Marie|Simpson|2014-05-09|  F|ET74 2SP|       Smart|
    |  5|  Maggie|      null|Simpson|2021-01-12|  F|ET74 2SP|   Star-hair|
    |  4|    Lisa|     Marie|Simpson|2014-05-09|  F|ET74 2SP|   Red-dress|
    |  3|    Bart|     Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|  Rebellious|
    |  2|   Marge|    Juliet|Simpson|1983-03-19|  F|ET74 2SP|   Blue-hair|
    |  2|   Marge|    Juliet|Simpson|1983-03-19|  F|ET74 2SP|Kind-hearted|
    |  5|  Maggie|      null|Simpson|2021-01-12|  F|ET74 2SP|        Mute|
    |  1|   Homer|       Jay|Simpson|1983-05-12|  M|ET74 2SP|        Lazy|
    +---+--------+----------+-------+----------+---+--------+------------+

    if you wanted to also keep the original overall description:
    > explode(df,column = 'Description',on = ' ',retain = True,drop_duplicates = True, flag = None).show()  
    +---+--------+----------+-------+----------+---+--------+----------------------+
    |ID |Forename|Middlename|Surname|DoB       |Sex|Postcode|Description           |
    +---+--------+----------+-------+----------+---+--------+----------------------+
    |3  |Bart    |Jo-Jo     |Simpson|2012-04-01|M  |ET74 2SP|Spikey-hair           |
    |1  |Homer   |Jay       |Simpson|1983-05-12|M  |ET74 2SP|Balding               |
    |4  |Lisa    |Marie     |Simpson|2014-05-09|F  |ET74 2SP|Smart                 |
    |5  |Maggie  |null      |Simpson|2021-01-12|F  |ET74 2SP|Star-hair             |
    |4  |Lisa    |Marie     |Simpson|2014-05-09|F  |ET74 2SP|Red-dress             |
    |3  |Bart    |Jo-Jo     |Simpson|2012-04-01|M  |ET74 2SP|Rebellious            |
    |2  |Marge   |Juliet    |Simpson|1983-03-19|F  |ET74 2SP|Blue-hair             |
    |1  |Homer   |Jay       |Simpson|1983-05-12|M  |ET74 2SP|Balding Lazy          |
    |2  |Marge   |Juliet    |Simpson|1983-03-19|F  |ET74 2SP|Kind-hearted          |
    |3  |Bart    |Jo-Jo     |Simpson|2012-04-01|M  |ET74 2SP|Spikey-hair Rebellious|
    |5  |Maggie  |null      |Simpson|2021-01-12|F  |ET74 2SP|Star-hair Mute        |
    |5  |Maggie  |null      |Simpson|2021-01-12|F  |ET74 2SP|Mute                  |
    |1  |Homer   |Jay       |Simpson|1983-05-12|M  |ET74 2SP|Lazy                  |
    |2  |Marge   |Juliet    |Simpson|1983-03-19|F  |ET74 2SP|Blue-hair Kind-hearted|
    |4  |Lisa    |Marie     |Simpson|2014-05-09|F  |ET74 2SP|Red-dress Smart       |
    +---+--------+----------+-------+----------+---+--------+----------------------+
    """

    if retain == False:

        df = (df
              .where(F.col(column).rlike(on))
              .select(*[x for x in df.columns if x != column],
                      F.explode(F.split(F.col(column), on))
                      .alias(column))
              .unionByName((df
                           .where((F.col(column).rlike(on) == False)
                                  | (F.col(column).rlike(on).isNull()))))
              )

    if retain == True:

        if flag is None:

            df = (df
                  .where(F.col(column).rlike(on))
                  .select(*[x for x in df.columns if x != column],
                          F.explode(F.split(F.col(column), on))
                          .alias(column))
                  .unionByName(df)
                  )

        else:

            df = (df
                  .where(F.col(column).rlike(on))
                  .withColumn(flag, F.lit(True))
                  .select(*[x for x in df.columns+[flag] if x != column],
                          F.explode(F.split(F.col(column), on))
                          .alias(column))
                  .unionByName(df.withColumn(flag, F.lit(False)))
                  )

    if drop_duplicates == True:
        df = df.dropDuplicates()

    return df


#############################################################################

def rename_columns(df, rename_dict={}):
    """
    Allows multiple columns to be renamed in one command from {before:after} 
    replacement dictionary

    Parameters
    ----------
    df : dataframe
      Dataframe to which the function is applied.
    rename_dict : dictionary
      The dictionary to rename columns, with format
      before:after

    Returns
    -------
    dataframe
      Dataframe with columns renamed

    Raises:
    -------
    None at present

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

    > rename_columns(df,rename_dict = {'ID':'Number','DoB':'Birth Date'}).show()
    +------+--------+----------+-------+----------+---+--------+
    |Number|Forename|Middlename|Surname|Birth Date|Sex|Postcode|
    +------+--------+----------+-------+----------+---+--------+
    |     1|   Homer|       Jay|Simpson|1983-05-12|  M|ET74 2SP|
    |     2|   Marge|    Juliet|Simpson|1983-03-19|  F|ET74 2SP|
    |     3|    Bart|     Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|
    |     3|    Bart|     Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|
    |     4|    Lisa|     Marie|Simpson|2014-05-09|  F|ET74 2SP|
    |     5|  Maggie|      null|Simpson|2021-01-12|  F|ET74 2SP|
    +------+--------+----------+-------+----------+---+--------+

    """

    for before, after in rename_dict.items():
        df = df.withColumnRenamed(before, after)

    return df

#############################################################################


def prefix_columns(df, prefix, exclude=[]):
    """
    Renames columns with specified prefix string.

    Parameters
    ----------
    df: dataframe
    prefix : string 
      The prifix string that will be appended
      to column names.  
    exclude : string or list of strings, default = None
      This argument either takes a list of column names
      or a string value that is a column name. 
      These values are excluded from the renaming of
      columns.

    Returns
    -------
    dataframe
      Dataframe with prefixed column names.

    Raises:
    -------
    None at present.

    Example
    -------
    e.g., you want to join the Simpsons df to the Flintstones df, 
    suffixing or prefixing will allow you to identify which data set the columns
    relate to:

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

    > prefix_columns(df,prefix='Simpsons_').show()
    +-----------+-----------------+-------------------+----------------+------------+------------+-----------------+
    |Simpsons_ID|Simpsons_Forename|Simpsons_Middlename|Simpsons_Surname|Simpsons_DoB|Simpsons_Sex|Simpsons_Postcode|
    +-----------+-----------------+-------------------+----------------+------------+------------+-----------------+
    |          1|            Homer|                Jay|         Simpson|  1983-05-12|           M|         ET74 2SP|
    |          2|            Marge|             Juliet|         Simpson|  1983-03-19|           F|         ET74 2SP|
    |          3|             Bart|              Jo-Jo|         Simpson|  2012-04-01|           M|         ET74 2SP|
    |          3|             Bart|              Jo-Jo|         Simpson|  2012-04-01|           M|         ET74 2SP|
    |          4|             Lisa|              Marie|         Simpson|  2014-05-09|           F|         ET74 2SP|
    |          5|           Maggie|               null|         Simpson|  2021-01-12|           F|         ET74 2SP|
    +-----------+-----------------+-------------------+----------------+------------+------------+-----------------+

    > prefix_columns(df,prefix='Simpsons_',exclude='Surname').show()
    +-----------+-----------------+-------------------+-------+------------+------------+-----------------+
    |Simpsons_ID|Simpsons_Forename|Simpsons_Middlename|Surname|Simpsons_DoB|Simpsons_Sex|Simpsons_Postcode|
    +-----------+-----------------+-------------------+-------+------------+------------+-----------------+
    |          1|            Homer|                Jay|Simpson|  1983-05-12|           M|         ET74 2SP|
    |          2|            Marge|             Juliet|Simpson|  1983-03-19|           F|         ET74 2SP|
    |          3|             Bart|              Jo-Jo|Simpson|  2012-04-01|           M|         ET74 2SP|
    |          3|             Bart|              Jo-Jo|Simpson|  2012-04-01|           M|         ET74 2SP|
    |          4|             Lisa|              Marie|Simpson|  2014-05-09|           F|         ET74 2SP|
    |          5|           Maggie|               null|Simpson|  2021-01-12|           F|         ET74 2SP|
    +-----------+-----------------+-------------------+-------+------------+------------+-----------------+

    """

    if type(exclude) != list:
        exclude = [exclude]

    old = [x for x in df.columns if x not in exclude]
    new = [prefix + x for x in old]

    rename = dict(zip(old, new))

    for old, new in rename.items():
        df = df.withColumnRenamed(old, new)

    return df

#############################################################################


def suffix_columns(df, suffix, exclude=[]):
    """
    Renames columns with specified suffix string.

    Parameters
    ----------
    df: dataframe
    suffix : string 
      The suffix string that will be appended
      to column names. 
    exclude : string or list of strings
      This argument either takes a list of column names
      or a string value that is a column name. 
      These values are excluded from the renaming of
      columns.

    Returns
    -------
    dataframe
      Dataframe with suffixed column names.

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

    > suffix_columns(df,suffix='_Simpsons').show()
    +-----------+-----------------+-------------------+----------------+------------+------------+-----------------+
    |ID_Simpsons|Forename_Simpsons|Middlename_Simpsons|Surname_Simpsons|DoB_Simpsons|Sex_Simpsons|Postcode_Simpsons|
    +-----------+-----------------+-------------------+----------------+------------+------------+-----------------+
    |          1|            Homer|                Jay|         Simpson|  1983-05-12|           M|         ET74 2SP|
    |          2|            Marge|             Juliet|         Simpson|  1983-03-19|           F|         ET74 2SP|
    |          3|             Bart|              Jo-Jo|         Simpson|  2012-04-01|           M|         ET74 2SP|
    |          3|             Bart|              Jo-Jo|         Simpson|  2012-04-01|           M|         ET74 2SP|
    |          4|             Lisa|              Marie|         Simpson|  2014-05-09|           F|         ET74 2SP|
    |          5|           Maggie|               null|         Simpson|  2021-01-12|           F|         ET74 2SP|
    +-----------+-----------------+-------------------+----------------+------------+------------+-----------------+

    > suffix_columns(df,suffix='_Simpsons',exclude='Surname').show()
    +-----------+-----------------+-------------------+-------+------------+------------+-----------------+
    |ID_Simpsons|Forename_Simpsons|Middlename_Simpsons|Surname|DoB_Simpsons|Sex_Simpsons|Postcode_Simpsons|
    +-----------+-----------------+-------------------+-------+------------+------------+-----------------+
    |          1|            Homer|                Jay|Simpson|  1983-05-12|           M|         ET74 2SP|
    |          2|            Marge|             Juliet|Simpson|  1983-03-19|           F|         ET74 2SP|
    |          3|             Bart|              Jo-Jo|Simpson|  2012-04-01|           M|         ET74 2SP|
    |          3|             Bart|              Jo-Jo|Simpson|  2012-04-01|           M|         ET74 2SP|
    |          4|             Lisa|              Marie|Simpson|  2014-05-09|           F|         ET74 2SP|
    |          5|           Maggie|               null|Simpson|  2021-01-12|           F|         ET74 2SP|
    +-----------+-----------------+-------------------+-------+------------+------------+-----------------+

    """

    if type(exclude) != list:
        exclude = [exclude]

    old = [x for x in df.columns if x not in exclude]
    new = [x+suffix for x in old]

    rename = dict(zip(old, new))

    for old, new in rename.items():
        df = df.withColumnRenamed(old, new)

    return df

#############################################################################


def union_all(*dfs, fill=None):
    """
    Unions list of dataframes to a single dataframe. Where dataframe columns
    are not consistent, creates columns to enable union with default null fill

    Parameters
    ----------
    *dfs : *args
      an optional length list of dataframes to be combined
    fill : some kind of data-type, default = None
      optional value to fill null values with when column names are inconsistent
      between the dataframes being combined

    Returns
    -------
    dataframe
      Single unioned dataframe. 

    Raises
    ------
    None at present.

    Example
    -------

    > df1.show()

    +---+--------+----------+-------+----------+---+------------------------+
    |ID |Forename|Middlename|Surname|DoB       |Sex|Profession              |
    +---+--------+----------+-------+----------+---+------------------------+
    |1  |Homer   |Jay       |Simpson|1983-05-12|M  |Nuclear safety inspector|
    |2  |Marge   |Juliet    |Simpson|1983-03-19|F  |Housewife               |
    +---+--------+----------+-------+----------+---+------------------------+

    > df2.show()

    +---+--------+----------+-------+----------+---+
    | ID|Forename|Middlename|Surname|       DoB|Sex|
    +---+--------+----------+-------+----------+---+
    |  3|    Bart|     Jo-Jo|Simpson|2012-04-01|  M|
    |  4|    Lisa|     Marie|Simpson|2014-05-09|  F|
    |  5|  Maggie|      null|Simpson|2021-01-12|  F|
    +---+--------+----------+-------+----------+---+

    > union_all(df1,df2).show(truncate = False)

    +---+--------+----------+-------+----------+---+------------------------+
    |ID |Forename|Middlename|Surname|DoB       |Sex|Profession              |
    +---+--------+----------+-------+----------+---+------------------------+
    |1  |Homer   |Jay       |Simpson|1983-05-12|M  |Nuclear safety inspector|
    |2  |Marge   |Juliet    |Simpson|1983-03-19|F  |Housewife               |
    |3  |Bart    |Jo-Jo     |Simpson|2012-04-01|M  |null                    |
    |4  |Lisa    |Marie     |Simpson|2014-05-09|F  |null                    |
    |5  |Maggie  |null      |Simpson|2021-01-12|F  |null                    |
    +---+--------+----------+-------+----------+---+------------------------+

    > union_all(df1,df2, fill = 'too young').show(truncate = False)

    +---+--------+----------+-------+----------+---+------------------------+
    |ID |Forename|Middlename|Surname|DoB       |Sex|Profession              |
    +---+--------+----------+-------+----------+---+------------------------+
    |1  |Homer   |Jay       |Simpson|1983-05-12|M  |Nuclear safety inspector|
    |2  |Marge   |Juliet    |Simpson|1983-03-19|F  |Housewife               |
    |3  |Bart    |Jo-Jo     |Simpson|2012-04-01|M  |too young               |
    |4  |Lisa    |Marie     |Simpson|2014-05-09|F  |too young               |
    |5  |Maggie  |null      |Simpson|2021-01-12|F  |too young               |
    +---+--------+----------+-------+----------+---+------------------------+

    """

    if len(dfs) == 1:
        return dfs[0]

    columns = list(set([x for y in
                        [df.columns for df in dfs]
                        for x in y]))

    out = dfs[0]

    add_columns = [x for x in columns
                   if x not in out.columns]

    for col in add_columns:
        out = out.withColumn(col, F.lit(fill))

    for df in dfs[1:]:

        add_columns = [x for x in columns
                       if x not in df.columns]

        for col in add_columns:
            df = df.withColumn(col, F.lit(fill))

        out = out.unionByName(df)

    return out

#############################################################################


def drop_nulls(df, subset=None, val=None):
    """
    drop_nulls can drop either rows with Null values or values that are 
    specified.

    This drops rows containing nulls in any columns by default.

    Parameters
    ----------
    df : dataframe
      The dataframe the function is applied to.  
    subset : string or list of strings, default = None
      A list of columns to drop null values from
    val : string, default = None
      The specified value for nulls

    Returns
    -------
    dataframe
      Dataframe with Null/ val values dropped on
      the columns where the function is applied.

    Raises
    ------
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
    |  4|    Lisa|      null|Simpson|2014-05-09|  F|ET74 2SP|
    |  5|  Maggie|      null|Simpson|2021-01-12|  F|ET74 2SP|
    +---+--------+----------+-------+----------+---+--------+

    > drop_nulls(df, subset = None, val = None).show()

    +---+--------+----------+-------+----------+---+--------+
    | ID|Forename|Middlename|Surname|       DoB|Sex|Postcode|
    +---+--------+----------+-------+----------+---+--------+
    |  3|    Bart|     Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|
    +---+--------+----------+-------+----------+---+--------+

    """

    if subset is not None:

        if type(subset) != list:
            subset = [subset]

    if val != None:
        df = df.replace(val, value=None, subset=subset)

    df = df.dropna(how='any', subset=subset)

    return df

###############################################################################


def window(df, window, target, mode, alias=None, drop_na=False):
    """
    Adds window column for count, countDistinct, min, max, or sum operations 
    over window
    
    Need to import the union_all function first.

    Parameters
    ----------
    df : dataframe
      Dataframe to which the function is applied.
    window : string or list of strings
      List of columns defining the window.
    target : string
      Name of target column for operations in 
      string format.
    mode : {'count','countDistinct','min','max','sum'}
      Operation performed on window.
    alias : string, default = None
      Name of column for window function results.
    drop_na : bool, default = False
      drops Null/NA values from countDistinct
      window function when performing the 
      operation.

    Returns
    -------
    dataframe
      Dataframe with window alias column appended
      to dataframe showing results of the
      operation performed over a window of columns.

    Raises
    ------
    None at present.

    Example
    -------

    > df.show()

    +---+--------+----------+-------+----------+---+--------+-----------------+
    | ID|Forename|Middlename|Surname|       DoB|Sex|Postcode|age_at_2022-12-06|
    +---+--------+----------+-------+----------+---+--------+-----------------+
    |  1|   Homer|       Jay|Simpson|1983-05-12|  M|ET74 2SP|               39|
    |  2|   Marge|    Juliet|Simpson|1983-03-19|  F|ET74 2SP|               39|
    |  3|    Bart|     Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|               10|
    |  3|    Bart|     Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|               10|
    |  4|    Lisa|     Marie|Simpson|2014-05-09|  F|ET74 2SP|                8|
    |  5|  Maggie|      null|Simpson|2021-01-12|  F|ET74 2SP|                1|
    +---+--------+----------+-------+----------+---+--------+-----------------+

    > window(df = df,
             window = 'ID',
             target = 'Forename',
             mode = 'count',
             alias= 'forenames_per_ID',
             drop_na=False).show()

    +---+--------+----------+-------+----------+---+--------+-----------------+----------------+
    | ID|Forename|Middlename|Surname|       DoB|Sex|Postcode|age_at_2022-12-06|forenames_per_ID|
    +---+--------+----------+-------+----------+---+--------+-----------------+----------------+
    |  3|    Bart|     Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|               10|               2|
    |  3|    Bart|     Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|               10|               2|
    |  5|  Maggie|      null|Simpson|2021-01-12|  F|ET74 2SP|                1|               1|
    |  1|   Homer|       Jay|Simpson|1983-05-12|  M|ET74 2SP|               39|               1|
    |  4|    Lisa|     Marie|Simpson|2014-05-09|  F|ET74 2SP|                8|               1|
    |  2|   Marge|    Juliet|Simpson|1983-03-19|  F|ET74 2SP|               39|               1|
    +---+--------+----------+-------+----------+---+--------+-----------------+----------------+

    > window(df = df,
             window = 'ID',
             target = 'Forename',
             mode = 'countDistinct',
             alias= 'distinct_forenames_per_ID',
             drop_na=False).show()
             
    +---+-------------------------+--------+----------+-------+----------+---+--------+-----------------+
    | ID|distinct_forenames_per_ID|Forename|Middlename|Surname|       DoB|Sex|Postcode|age_at_2022-12-06|
    +---+-------------------------+--------+----------+-------+----------+---+--------+-----------------+
    |  3|                        1|    Bart|     Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|               10|
    |  3|                        1|    Bart|     Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|               10|
    |  5|                        1|  Maggie|      null|Simpson|2021-01-12|  F|ET74 2SP|                1|
    |  1|                        1|   Homer|       Jay|Simpson|1983-05-12|  M|ET74 2SP|               39|
    |  4|                        1|    Lisa|     Marie|Simpson|2014-05-09|  F|ET74 2SP|                8|
    |  2|                        1|   Marge|    Juliet|Simpson|1983-03-19|  F|ET74 2SP|               39|
    +---+-------------------------+--------+----------+-------+----------+---+--------+-----------------+

    > window(df = df,
             window = 'Sex',
             target = 'age_at_2022-12-09',
             mode = 'min',
             alias= 'youngest_per_sex',
             drop_na=False).show()
             
    +---+----------------+---+--------+----------+-------+----------+--------+-----------------+
    |Sex|youngest_per_sex| ID|Forename|Middlename|Surname|       DoB|Postcode|age_at_2022-12-09|
    +---+----------------+---+--------+----------+-------+----------+--------+-----------------+
    |  F|               1|  2|   Marge|    Juliet|Simpson|1983-03-19|ET74 2SP|               39|
    |  F|               1|  4|    Lisa|     Marie|Simpson|2014-05-09|ET74 2SP|                8|
    |  F|               1|  5|  Maggie|      null|Simpson|2021-01-12|ET74 2SP|                1|
    |  M|              10|  1|   Homer|       Jay|Simpson|1983-05-12|ET74 2SP|               39|
    |  M|              10|  3|    Bart|     Jo-Jo|Simpson|2012-04-01|ET74 2SP|               10|
    |  M|              10|  3|    Bart|     Jo-Jo|Simpson|2012-04-01|ET74 2SP|               10|
    +---+----------------+---+--------+----------+-------+----------+--------+-----------------+

    > window(df = df,
             window = 'Sex',
             target = 'age_at_2022-12-09',
             mode = 'max',
             alias= 'oldest_per_sex',
             drop_na=False).show()

    +---+--------------+---+--------+----------+-------+----------+--------+-----------------+
    |Sex|oldest_per_sex| ID|Forename|Middlename|Surname|       DoB|Postcode|age_at_2022-12-09|
    +---+--------------+---+--------+----------+-------+----------+--------+-----------------+
    |  F|            39|  2|   Marge|    Juliet|Simpson|1983-03-19|ET74 2SP|               39|
    |  F|            39|  4|    Lisa|     Marie|Simpson|2014-05-09|ET74 2SP|                8|
    |  F|            39|  5|  Maggie|      null|Simpson|2021-01-12|ET74 2SP|                1|
    |  M|            39|  3|    Bart|     Jo-Jo|Simpson|2012-04-01|ET74 2SP|               10|
    |  M|            39|  1|   Homer|       Jay|Simpson|1983-05-12|ET74 2SP|               39|
    |  M|            39|  3|    Bart|     Jo-Jo|Simpson|2012-04-01|ET74 2SP|               10|
    +---+--------------+---+--------+----------+-------+----------+--------+-----------------+

    > window(df = df,
             window = 'Sex',
             target = 'age_at_2022-12-09',
             mode = 'sum',
             alias= 'total_age_by_sex',
             drop_na=False).show()
             
    +---+----------------+---+--------+----------+-------+----------+--------+-----------------+
    |Sex|total_age_by_sex| ID|Forename|Middlename|Surname|       DoB|Postcode|age_at_2022-12-09|
    +---+----------------+---+--------+----------+-------+----------+--------+-----------------+
    |  F|              48|  2|   Marge|    Juliet|Simpson|1983-03-19|ET74 2SP|               39|
    |  F|              48|  4|    Lisa|     Marie|Simpson|2014-05-09|ET74 2SP|                8|
    |  F|              48|  5|  Maggie|      null|Simpson|2021-01-12|ET74 2SP|                1|
    |  M|              59|  1|   Homer|       Jay|Simpson|1983-05-12|ET74 2SP|               39|
    |  M|              59|  3|    Bart|     Jo-Jo|Simpson|2012-04-01|ET74 2SP|               10|
    |  M|              59|  3|    Bart|     Jo-Jo|Simpson|2012-04-01|ET74 2SP|               10|
    +---+----------------+---+--------+----------+-------+----------+--------+-----------------+  
             
    See Also
    --------
    standardisation.standardise_null()
    """

    if type(window) != list:
        window = [window]

    w = Window.partitionBy(window)

    if mode == 'count':

        if alias is not None:

            df = (df
                  .select(*df.columns,
                          F.count(target)
                          .over(w)
                          .alias(alias))
                  )

        else:
            df = (df
                  .select(*df.columns,
                          F.count(target)
                          .over(w))
                  )

    if mode == 'countDistinct':

        df = df.fillna("<<<>>>", subset=window)

        if alias is not None:
            if drop_na == True:
                df = (df
                      .dropDuplicates(subset=window+[target])
                      .dropna(subset=[target])
                      .select(*window+[target],
                              F.count(target)
                              .over(w)
                              .alias(alias))
                      .drop(target)
                      .dropDuplicates()
                      ).join(df,
                             on=window,
                             how='right')
            else:
                df = (df
                      .dropDuplicates(subset=window+[target])
                      .select(*window+[target],
                              F.count(target)
                              .over(w)
                              .alias(alias))
                      .drop(target)
                      .dropDuplicates()
                      ).join(df,
                             on=window,
                             how='right')

        else:
            if drop_na == True:
                df = (df
                      .dropDuplicates(subset=window+[target])
                      .dropna(subset=[target])
                      .select(*window+[target],
                              F.count(target)
                              .over(w))
                      .drop(target)
                      .dropDuplicates()
                      ).join(df,
                             on=window,
                             how='right')
            else:
                df = (df
                      .dropDuplicates(subset=window+[target])
                      .select(*window+[target],
                              F.count(target)
                              .over(w))
                      .drop(target)
                      .dropDuplicates()
                      ).join(df,
                             on=window,
                             how='right')


    if mode == 'min':
        if alias is not None:

            df_1 = (df
                    .dropna(subset=target)
                    .select(*df.columns,
                            F.min(target)
                            .over(w)
                            .alias(alias))
                    )

            df_2 = (df
                    .where((F.col(target).isNull())
                           | F.isnan(F.col(target)))
                    .join(df_1.select(window),
                          on=window,
                          how='left_anti')
                    )

            df = (union_all(df_1, df_2)
                  .select(window+[alias])
                  .dropDuplicates()
                  .join(df,
                        on=window,
                        how='right')
                  )

        else:
            df_1 = (df
                    .dropna(subset=target)
                    .select(*df.columns,
                            F.min(target)
                            .over(w))
                    )

            df_2 = (df
                    .where((F.col(target).isNull())
                           | F.isnan(F.col(target)))
                    .join(df_1.select(window),
                          on=window,
                          how='left_anti')
                    )

            df = (union_all(df_1, df_2)
                  .drop(*[x for x in df.columns
                        if x not in window])
                  .dropDuplicates()
                  .join(df,
                        on=window,
                        how='right')
                  )

    if mode == 'max':
        if alias is not None:

            df_1 = (df
                    .dropna(subset=target)
                    .select(*df.columns,
                            F.max(target)
                            .over(w)
                            .alias(alias))
                    .select(window+[alias])
                    )

            df_2 = (df
                    .where((F.col(target).isNull())
                           | F.isnan(F.col(target)))
                    .join(df_1.select(window),
                          on=window,
                          how='left_anti')
                    .select(window)
                    )

            df = (union_all(df_1,df_2)
                  .select(window+[alias])
                  .dropDuplicates()
                  .join(df,
                        on=window,
                        how='right')
                  )

        else:

            # needs alternative to selecting alias

            drop = [x for x in df.columns
                    if x not in window]

            df_1 = (df
                    .dropna(subset=target)
                    .select(*df.columns,
                            F.max(target)
                            .over(w))
                    .drop(*drop)
                    )

            df_2 = (df
                    .where((F.col(target).isNull())
                           | F.isnan(F.col(target)))
                    .join(df_1.select(window),
                          on=window,
                          how='left_anti')
                    .select(window)
                    )

            df = (union_all(df_1, df_2)
                  .drop(*[x for x in df.columns
                        if x not in window])
                  .dropDuplicates()
                  .join(df,
                        on=window,
                        how='right')
                  )

    if mode == 'sum':
        if alias is not None:

            df_1 = (df
                    .dropna(subset=target)
                    .select(*df.columns,
                            F.sum(target)
                            .over(w)
                            .alias(alias))
                    .select(window+[alias])
                    )

            df_2 = (df
                    .where((F.col(target).isNull())
                           | F.isnan(F.col(target)))
                    .join(df_1.select(window),
                          on=window,
                          how='left_anti')
                    .select(window)
                    )

            df = (union_all(df_1, df_2)
                  .select(window+[alias])
                  .dropDuplicates()
                  .join(df,
                        on=window,
                        how='right')
                  )

        else:

            # needs alternative to selecting alias

            drop = [x for x in df.columns
                    if x not in window]

            df_1 = (df
                    .dropna(subset=target)
                    .select(*df.columns,
                            F.sum(target)
                            .over(w))
                    .drop(*drop)
                    )

            df_2 = (df
                    .where((F.col(target).isNull())
                           | F.isnan(F.col(target)))
                    .join(df_1.select(window),
                          on=window,
                          how='left_anti')
                    .select(window)
                    )

            df = (union_all(df_1, df_2)
                  .drop(*[x for x in df.columns
                        if x not in window])
                  .dropDuplicates()
                  .join(df,
                        on=window,
                        how='right')
                  )

    df = st.standardise_null(df, "^<<<>>>$", subset=window)

    return df


###############################################################################
def filter_window(df, filter_window, target, mode, value=None, condition=True):
    """  
    Performs statistical operations such as count, countDistinct, min or max on a
    collection of rows and returns results for each row individually.

    This function filters the results of the window operation in two ways;
    for count and count distinct, it filters the results to show the results
    where the 'value' argument value is matched. For min and max operations
    it filters window results to only the minimum or maximum values respectively.

    Parameters
    ----------
    df : dataframe
      Dataframe to which the function is applied.
    filter_window : string or list of strings
      List of columns defining a window
    target : string
      Target column for operations.
    mode : {'count','countDistinct'}
      Operation applied to the window.
    value : int, default = None
      a value to filter the data by after applying the window operation
    condition : bool, default = True
      option to include (True) or exclude (False) rows that match the
      filter value

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

    > filter_window(df = df,
                    filter_window = 'Forename',
                    target = 'ID',
                    mode = 'count',
                    value=1,
                    condition=True).show()

    +---+--------+----------+-------+----------+---+--------+
    | ID|Forename|Middlename|Surname|       DoB|Sex|Postcode|
    +---+--------+----------+-------+----------+---+--------+
    |  1|   Homer|       Jay|Simpson|1983-05-12|  M|ET74 2SP|
    |  2|   Marge|    Juliet|Simpson|1983-03-19|  F|ET74 2SP|
    |  5|  Maggie|      null|Simpson|2021-01-12|  F|ET74 2SP|
    |  4|    Lisa|     Marie|Simpson|2014-05-09|  F|ET74 2SP|
    +---+--------+----------+-------+----------+---+--------+

    > filter_window(df = df,
                    filter_window = 'Forename',
                    target = 'ID',
                    mode = 'count',
                    value=1,
                    condition=False).show()

    +---+--------+----------+-------+----------+---+--------+
    | ID|Forename|Middlename|Surname|       DoB|Sex|Postcode|
    +---+--------+----------+-------+----------+---+--------+
    |  3|    Bart|     Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|
    |  3|    Bart|     Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|
    +---+--------+----------+-------+----------+---+--------+
    
    >df.show()
    +---+---+----------+-------+----------+---+--------+
    |Age| ID|Middlename|Surname|       DoB|Sex|Postcode|
    +---+---+----------+-------+----------+---+--------+
    |  3|  2|    Juliet|Simpson|1983-03-19|  F|ET74 2SP|
    |  5|  4|     Marie|Simpson|2014-05-09|  F|ET74 2SP|
    |  6|  3|     Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|
    |  1|  5|      null|Simpson|2021-01-12|  F|ET74 2SP|
    |  4|  3|     Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|
    |  2|  1|       Jay|Simpson|1983-05-12|  M|ET74 2SP|
    +---+---+----------+-------+----------+---+--------+
    
    > filter_window(df = df,
                    filter_window = 'ID',
                    target = 'Age',
                    mode = 'min',
                    value=None,
                    condition=True).show()
    +---+---+----------+-------+----------+---+--------+
    | ID|Age|Middlename|Surname|       DoB|Sex|Postcode|
    +---+---+----------+-------+----------+---+--------+
    |  3|  4|     Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|
    |  5|  1|      null|Simpson|2021-01-12|  F|ET74 2SP|
    |  1|  2|       Jay|Simpson|1983-05-12|  M|ET74 2SP|
    |  4|  5|     Marie|Simpson|2014-05-09|  F|ET74 2SP|
    |  2|  3|    Juliet|Simpson|1983-03-19|  F|ET74 2SP|
    +---+---+----------+-------+----------+---+--------+


    See Also
    --------
    standardisation.fill_nulls()
    """

    if type(filter_window) != list:
        filter_window = [filter_window]

    w = Window.partitionBy(filter_window)

    if mode in ['count', 'countDistinct']:

        if condition:

            df = (window(df, filter_window, target, mode, alias='count')
                  .where(F.col('count') == value)
                  .drop('count')
                  )

        else:

            df = (window(df, filter_window, target, mode, alias='count')
                  .where(F.col('count') != value)
                  .drop('count')
                  )

    if mode in ['min','max']:

      if condition:

        df = window(df,filter_window,target,mode,alias='value')

        df = st.fill_nulls(df,fill='<<<>>>',subset=['value']+[target])

        df = (df
              .where(F.col(target)==F.col('value'))
              .drop('value')
             )

        df = (st.standardise_null(df = df,
                                  replace = "^<<<>>>$",
                                  subset = target)
           )

      else:

        df = window(df,filter_window,target,mode,alias='value')

        df = st.fill_nulls(df,fill='<<<>>>',subset=['value']+[target])

        df = (df
              .where(F.col(target)!=F.col('value'))
              .drop('value')
             )

        df = (st.standardise_null(df = df,
                                  replace = "^<<<>>>$",
                                  subset = target)
           )

    return df

###############################################################################


def literal_column(df, col_name, literal):
    """
    literal_column returns the original dataframe along with
    a new column added containing values specified by the user.

    Parameters
    ----------
    df : dataframe
      Dataframe to which the function is applied.
    col_name : string
      New column title.
    literal :data-type
      Values populating the colName column.

    Returns
    -------
    dataframe
      Dataframe with new literal column.

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

    > literal_column(df,col_name = 'Next-door neighbour',literal = 'Ned Flanders').show()
    +---+--------+----------+-------+----------+---+--------+-------------------+
    | ID|Forename|Middlename|Surname|       DoB|Sex|Postcode|Next-door neighbour|
    +---+--------+----------+-------+----------+---+--------+-------------------+
    |  1|   Homer|       Jay|Simpson|1983-05-12|  M|ET74 2SP|       Ned Flanders|
    |  2|   Marge|    Juliet|Simpson|1983-03-19|  F|ET74 2SP|       Ned Flanders|
    |  3|    Bart|     Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|       Ned Flanders|
    |  3|    Bart|     Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|       Ned Flanders|
    |  4|    Lisa|     Marie|Simpson|2014-05-09|  F|ET74 2SP|       Ned Flanders|
    |  5|  Maggie|      null|Simpson|2021-01-12|  F|ET74 2SP|       Ned Flanders|
    +---+--------+----------+-------+----------+---+--------+-------------------+
    """

    df = df.withColumn(col_name, F.lit(literal))
    return df

###########################################################################


def coalesced(df, subset=None, output_col="coalesced_col"):
    """
    Produces a new column from a supplied dataframe, that contains the first
    non-null value from each row.

    Parameters:
    ----------
    df : dataframe
      Dataframe to which the function is applied.
    subset : list of strings, default = None
      Subset of columns being coalesced together
      into a single column, if subset = None then
      subset = all columns in dataframe.
    output_col : string, default = 'coalesced_col'
      Name of the output column for results of
      the coalesced columns.

    Returns
    -------
    dataframe
      Dataframe with coalesced columns results
      appended to original dataframe in the 
      output_col arg column.

    Raises
    ------
    None at present.

    Example
    -------

    > df3.show()
    +----+----+
    |   a|   b|
    +----+----+
    |null|null|
    |   1|null|
    |null|   2|
    +----+----+

    > dataframes.coalesced(df3,subset = None, output_col="coalesced_col").show()

    +----+----+-------------+
    |   a|   b|coalesced_col|
    +----+----+-------------+
    |null|null|        null |
    |   1|null|           1 |
    |null|   2|           2 |
    +----+----+-------------+
    """

    if subset == None:
        subset = df.columns

    df = df.withColumn(output_col,
                       F.coalesce(*[F.col(x) for x in subset]))
    return df

##############################################################################


def split(df, col_in, col_out=None, split_on=' '):
    """
    Splits a string column to array on specified separator. 
    Option to return split column as new column or to split in place.

    Parameters
    ----------
    df : dataframe
      Dataframe to which the function is applied.
    col_in : string
      column to be split to array
    col_out, string, default = None
      output column for split strings, 
      default value makes the split happen in place,
    split_on : string, default = ' '
      string or regex separator for split

    Returns
    -------
    dataframe
      Dataframe with split array column.

    Raises
    ------
    None at present.
    """

    if col_out is None:

        df = df.withColumn(col_in, F.when((F.col(col_in).isNull())
                                          | (F.isnan(F.col(col_in))), None)
                           .otherwise(F.split(F.col(col_in), split_on)))

    else:

        df = df.withColumn(col_out, F.when((F.col(col_in).isNull())
                                           | (F.isnan(F.col(col_in))), None)
                           .otherwise(F.split(F.col(col_in), split_on)))

    return df


##############################################################################

def index_select(df, split_col, out_col, index, sep=' '):
    """
    Allows indices to be selected within a column of arrays
    and casts those values to a new column (out_col arg).

    Parameters
    ----------
    df : dataframe
      Dataframe to which the function is applied.
    split_col : string
      Column header of the split array column to be indexed
    out_col : string
      Column header of the output column of selected index
    index : int or tuple of int
      Index or indices of required element(s) being selected
    sep : string, default = ' '
      Separator when using a tuple to select more than one
      element using index selection.

    Returns
    -------
    dataframe
      Dataframe with new variable selected from split array column

    Raises
    ------
    None at present

    Example
    -------

    > df3.show()

    +------+----+
    |     a|   b|
    +------+----+
    |[1, 2]|null|
    |[4, 5]|null|
    |[7, 8]|   2|
    +------+----+

    > dataframes.index_select(df3, split_col = 'a', out_col = 'a_index_1',
                              index = 0, sep=' ').show()

    +------+----+---------+
    |     a|   b|a_index_1|
    +------+----+---------+
    |[1, 2]|null|        1|
    |[4, 5]|null|        4|
    |[7, 8]|   2|        7|
    +------+----+---------+
    """

    if type(index) == tuple:

        for i in range(index[1])[index[0]:]:

            df = df.withColumn(f'index_select_{i}', F.col(split_col)
                               .getItem(i))

        df = concat(df, out_col, sep,
                    [f'index_select_{i}'
                     for i in range(index[1])[index[0]:]])

        df = df.drop(*[f'index_select_{i}'
                     for i in range(index[1])[index[0]:]])

    else:

        if index >= 0:

            df = (df.
                  withColumn(out_col, F.col(split_col)
                             .getItem(index)))

        if index < 0:
            df = (df
                  .withColumn(out_col, F.reverse(F.col(split_col))
                              .getItem(abs(index)-1)))

    return df

#############################################################################


def clone_column(df, target, clone):
    """
    Clones a column within a dataframe and gives it
    a new column header.

    Parameters
    ----------
    df : dataframe
      Dataframe to which the function is applied.
    target : string
      The name of the column to be cloned.
    clone : string
      Name of the new column.

    Returns 
    -------
    dataframe
      Dataframe with column cloned.

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

    > clone_column(df, target = 'Sex', clone = 'Gender').show()
      +---+--------+----------+-------+----------+---+--------+------+
      | ID|Forename|Middlename|Surname|       DoB|Sex|Postcode|Gender|
      +---+--------+----------+-------+----------+---+--------+------+
      |  1|   Homer|       Jay|Simpson|1983-05-12|  M|ET74 2SP|     M|
      |  2|   Marge|    Juliet|Simpson|1983-03-19|  F|ET74 2SP|     F|
      |  3|    Bart|     Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|     M|
      |  3|    Bart|     Jo-Jo|Simpson|2012-04-01|  M|ET74 2SP|     M|
      |  4|    Lisa|     Marie|Simpson|2014-05-09|  F|ET74 2SP|     F|
      |  5|  Maggie|      null|Simpson|2021-01-12|  F|ET74 2SP|     F|
      +---+--------+----------+-------+----------+---+--------+------+

    """

    df = df.withColumn(clone, F.col(target))

    return df

############################################################


def substring(df, out_col, target_col,
              start, length, from_end=False):
    """
    Creates a new column containing substring values
    from another column. 

    Can either be a substring starting from the first character
    in the string if 'from_end' is False, or from the last
    character in the string if 'from_end' is true.

    Parameters
    ----------
    df : dataframe
      Dataframe to which the function is applied.
    out_col : string
      Column title for the new column which
      will store the substring values.
    target_col : string
      Column title for the target column to which
      the function is applied.
    start : int
      Index value of where the substring starts.
    length : int
      Length of substring being extracted to
      new column.
    from_end: bool, default = False
      Option to reverse the string before applying
      substring start and length arguments.

    Returns
    -------
    dataframe
      Returns dataframe with new column made up of
      substring values.

    Raises
    ------
    None at present.

    Example
    -------

    > df3.show()
    +--------+---+
    |       a|  b|
    +--------+---+
    |tomatoes|  b|
    |potatoes|  c|
    +--------+---+

    > dataframes.substring(df3,out_col = 'substring',target_col = 'a',
                  start = -2,length = 3,from_end=False).show()
    +--------+---+---------+
    |       a|  b|substring|
    +--------+---+---------+
    |tomatoes|  b|       es|
    |potatoes|  c|       es|
    +--------+---+---------+

    > dataframes.substring(df3,out_col = 'substring',target_col = 'a',
                  start = 2,length = 3,from_end=False).show()
    +--------+---+---------+
    |       a|  b|substring|
    +--------+---+---------+
    |tomatoes|  b|      oma|
    |potatoes|  c|      ota|
    +--------+---+---------+

    > dataframes.substring(df3,out_col = 'substring',target_col = 'a',
                  start = 2,length = 3,from_end=True).show()
    +--------+---+---------+
    |       a|  b|substring|
    +--------+---+---------+
    |tomatoes|  b|      toe|
    |potatoes|  c|      toe|
    +--------+---+---------+

    """

    if from_end == False:

        df = (df
              .withColumn(out_col,
                          F.substring(F.col(target_col),
                                      start, length))
              )

    if from_end == True:

        df = (df
              .withColumn(target_col,
                          F.reverse(F.col(target_col)))
              .withColumn(out_col,
                          F.reverse(F.substring(F.col(target_col),
                                                start, length)))
              .withColumn(target_col,
                          F.reverse(F.col(target_col)))
              )

    return df

################################################################################


def cut_off(df, threshold_column, val, mode):
    """
    cut_off cuts off rows that do not meet certain thresholds.

    cut_off takes a df column and a cutoff value and returns the dataframe
    with rows in which the mode and cutoff value condition is met.  
    cut_off will also work for date values if the threshold_column is a timestamp.

    Parameters
    ----------
    df : dataframe
      Dataframe to which the function is applied.
    threshold_column : string
      column to which the cutoff values are being applied.
    val : int or date or timestamp
      Value against which the mode operation is checking 
      threshold_column values.
    mode : string, {'<','<=','>','>='}
      Operation used to cutoff values that do not
      meet the operation requirement.

    Returns
    -------
    dataframe
      Dataframe with rows removed where they did
      not meet the cut off specification.

    Raises
    ------
    None at present.

    Example
    -------

    > df3.show()

    +---+---+
    |  a|  b|
    +---+---+
    |  1|  2|
    |100|200|
    +---+---+  

    > dataframes.cut_off(df3, threshold_column = 'a', val = 5, mode = '>').show()

    +---+---+
    |  a|  b|
    +---+---+
    |100|200|
    +---+---+

    """
    if mode == '>=':
        df = df.where(F.col(threshold_column) >= val)
    elif mode == '>':
        df = df.where(F.col(threshold_column) > val)
    elif mode == '<':
        df = df.where(F.col(threshold_column) < val)
    elif mode == '<=':
        df = df.where(F.col(threshold_column) <= val)
    return df

###############################################################################


def date_diff(df, col_name1, col_name2, diff='Difference',
             in_date_format='dd-mm-yyyy', units='days', absolute=True):
    """
    date_diff finds the number of days/months/years between two date columns
    by subtracting the dates in the second column from the dates in the first.
    Note, using just months is currently inaccurate as all months are assumed
    to have 31 days.

    Parameters
    ----------
    df : dataframe
      Dataframe to which the function is applied.
    col_name1 : string
      Name of the first column with values representing
      dates.
    col_name2 : string
      Name of second column with values representing
      dates.
    diff : string, default = 'Difference'
      Name of the column in which the difference between
      dates will be shown.
    in_date_format : string, default = 'dd-mm-yyyy'
      User must specify the format of how the dates are entered
      in both colName1 and colName2 and use this argument to
      do so.
    units : {'days','months','years'}
      units of how the difference in the two date columns
      will be represented in the 'diff' arg column.
    absolute : bool, default = True
      Bool toggle allowing user to display all values
      as absolute or non-absolute values in the 
      'diff' arg column.

    Returns
    -------
    dataframe
      Dataframe with new column appended showing
      the time difference between col_name1 and col_name2 
      columns in the units specified.

    Raises
    ------
    None at present.

    Example
    -------
    >df.show()

    +---+--------+-------+----------+---+--------+----------+
    | ID|Forename|Surname|       DoB|Sex|Postcode|     Today|
    +---+--------+-------+----------+---+--------+----------+
    |  1|   Homer|Simpson|1983-05-12|  M|ET74 2SP|2022-11-07|
    |  2|   Marge|Simpson|1983-03-19|  F|ET74 2SP|2022-11-07|
    |  3|    Bart|Simpson|2012-04-01|  M|ET74 2SP|2022-11-07|
    |  3|    Bart|Simpson|2012-04-01|  M|ET74 2SP|2022-11-07|
    |  4|    Lisa|Simpson|2014-05-09|  F|ET74 2SP|2022-11-07|
    |  5|  Maggie|Simpson|2021-01-12|  F|ET74 2SP|2022-11-07|
    +---+--------+-------+----------+---+--------+----------+

      > date_diff(df,
                'DoB',
                'Today',
                diff = 'Difference',
                in_date_format = 'yyy-MM-dd',
                units = 'days',
                absolute=True).show()

    +---+--------+-------+----------+---+--------+----------+----------+
    | ID|Forename|Surname|       DoB|Sex|Postcode|     Today|Difference|
    +---+--------+-------+----------+---+--------+----------+----------+
    |  1|   Homer|Simpson|1983-05-12|  M|ET74 2SP|2022-11-07|  14424.04|
    |  2|   Marge|Simpson|1983-03-19|  F|ET74 2SP|2022-11-07|   14478.0|
    |  3|    Bart|Simpson|2012-04-01|  M|ET74 2SP|2022-11-07|   3872.04|
    |  3|    Bart|Simpson|2012-04-01|  M|ET74 2SP|2022-11-07|   3872.04|
    |  4|    Lisa|Simpson|2014-05-09|  F|ET74 2SP|2022-11-07|   3104.04|
    |  5|  Maggie|Simpson|2021-01-12|  F|ET74 2SP|2022-11-07|     664.0|
    +---+--------+-------+----------+---+--------+----------+----------+

    """

    df = df.withColumn(diff, F.unix_timestamp(F.col(col_name1), in_date_format)
                       - F.unix_timestamp(F.col(col_name2), in_date_format))

    if units == 'days':
        df = df.withColumn(diff, (F.col(diff)/86400))
        df = df.withColumn(diff, F.round(diff, 2))
    elif units == 'months':
        # months value is slightly inaccurate as it assumes every month is a 31 day month
        df = df.withColumn(diff, F.col(diff)/(31*86400))
        df = df.withColumn(diff, F.round(diff, 2))
    elif units == 'years':
        df = df.withColumn(diff, F.col(diff)/(86400*365))
        df = df.withColumn(diff, F.round(diff, 2))
    if absolute == True:
        df = df.withColumn(diff, F.abs((F.col(diff))))
    return df
