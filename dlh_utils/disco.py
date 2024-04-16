'''Functions to help disclosure control code 
  Authors: Hannah Goode & Nathan O'Connor'''

import os
import random
import pandas as pd
import numpy as np

import pyspark
from pyspark.sql import SparkSession, Window
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, FloatType
import pyspark.sql.functions
from pyspark.sql.functions import regexp_extract, col, lit
from pyspark.sql.functions import concat, substring, expr, length
from pyspark.sql.functions import *
from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext

pd.set_option("display.max_rows", None)
pd.set_option('display.max_colwidth', None)

#########################---- datasets START -----##########################
'''
data = [("a1", "QQ 123456 C", "SPORTY", "PO356TH"),
        ("a2", "07451224152", "SCARY", "KZ66 7YT"),
        ("a3", "19760424", "BABY", "BH242ED"),
        ("a4", "19750127", "GINGER", "KN231SD"),
        ("a5", "19730724", "POSH", "HK192YC")]

schema = StructType([
    StructField("id", StringType(), True),
    StructField("dob", StringType(), True),
    StructField("name", StringType(), True),
    StructField("postcode", StringType(), True)
])

data = [("a1", "QQ 123456 C", "SPORTY", "PO356TH"),
("a2", "19830914", "SCARY", "KZ66 ZYT"),
("a3", "19760424", "BABY", "B242ED"),
("a4", "19750127", "GINGER", "KN231SD"),
("a5", "19730724", "POSH", "HK192YC"),
("ABE46372837", "19631011", "SKINNER", "SP63HW"),
("a7", "19780306", "FLANDERS", "6878844321"),
("a8", "07451224152", "BARNEY", "PO29 2YC"),
("a9", "19630507", "KRUSTY", "NP65LN"),
("a10", "19981218", "MILHOUSE", "MK17 6PO")
       ]

schema = StructType([
    StructField("id", StringType(), True),
    StructField("dob", StringType(), True),
    StructField("name", StringType(), True),
    StructField("postcode", StringType(), True)
])

df = spark.createDataFrame(data=data, schema=schema)
'''
#########################---- datasets END -----##########################

'''
tup = zip(random.sample(range(1,50),3)

          + random.sample(range(300,700),350)

          + random.sample(range(950,1000),3))


df_nums = spark.createDataFrame(data=tup,
                                schema=StructType([StructField("number",IntegerType(), True)]))

df_nums = df_nums.withColumn('id',
                             concat(lit('id'),
                                    row_number().over(Window.orderBy(lit(1)))))
'''
#########################---- regex table -----##########################

regex_data = [['NHS No','^\d{10}$','de_utils', ''],
              ['NIN','^\s*[a-zA-Z]{2}(?:\s*\d\s*){6}[a-dA-D]{1}?\s*$','Stack Overflow', ''],
              ['UK phone no','"^(?:0|\+?44)(?:\d\s?){9,10}$"','RegExLib.com', ''],
              ['UK Child Benefit','^[A-Z]{3}\d{8}$','RegExLib.com', ''],
              ['UK Passport','"[0-9]{10}[a-z]{3}[0-9]{7}[U,M,F]{1}[0-9]{9}"','RegExLib.com', ''],
              ['UK VAT','^\d{9}$','de-utils', ''],
              ['UK Vehicle Reg','^([A-Z]{3}\s?(\d{3}|\d{2}|d{1})\s?[A-Z])|([A-Z]\s?(\d{3}|\d{2}|\d{1})\s?[A-Z]{3})|(([A-HK-PRSVWY][A-HJ-PR-Y])\s?([0][2-9]|[1-9][0-9])\s?[A-HJ-PR-Z]{3})$','RegExLib.com', ''],
              ['UPRN','^[0-9]{12}$','Stack Overflow'],
              ['email_1','"^[^\.\s][\w\-\.{2,}]+@([\w-]+\.)+[\w-]{2,}$"','https://regex101.com/r/O9oCj8/1','Check for an e-mail'],
              ['filepath_1','/^(?:[\w]\:|\/)(\/[a-z_\-\s0-9\.]+)+\.(txt|gif|pdf|doc|docx|xls|xlsx|js)$','https://regex101.com/r/ZaU81J/1,Check for a file path'],
              ['free_text_1','[a-zA-Z0-9\s]','',"Check if it's a free text field. Any number of words having any number of spaces"]]

regex_cols = ['Identifier','Regex','Origin','Notes']

###############################################################################


def date_year_quarter(df, date_column):
    """
    Deletes your date column and replaces it with
    a year column and a year quarter column.

    The column can be string, integer or datestamp.

    The function will output 'input_error' if
    the length of numbers in the date_column is not equal to 8
    or the month in the date is not 01-12.

    prerequisites
    -------------
    Please insure that the date_column is fully numeric and the
    order is year then month then day e.g yyyyMMdd, yyyy-MM-dd etc...
    Please insure that the month is represented as two numbers
    i.e Janurary would be represented at 01 instead of 1

      Parameters
    ----------
    df: dataframe
      Dataframe to which the function is applied.
    date_column: date column (str)
      The name of the date column that you want to change to year and year quarter.

    Returns
    -------
    The function returns your dataframe but with the date column omitted and
    replaced with a date year and a date year quarter column.

    Example
    -------

    > df.show()
    +---+-----------+------+--------+
    | id|        dob|  name|postcode|
    +---+-----------+------+--------+
    | a1|QQ 123456 C|SPORTY| PO356TH|
    | a2|07451224152| SCARY|KZ66 ZYT|
    | a3|   19760424|  BABY| BH242ED|
    | a4|   19750127|GINGER| KN231SD|
    | a5|   19730724|  POSH| HK192YC|
    +---+-----------+------+--------+

    > date_year_quarter(df, date_column='dob').show()

    +---+------+--------+-----------+----------------+
    | id|  name|postcode|   dob_year|dob_year_quarter|
    +---+------+--------+-----------+----------------+
    | a1|SPORTY| PO356TH|input_error|     input_error|
    | a2| SCARY|KZ66 ZYT|input_error|     input_error|
    | a3|  BABY| BH242ED|       1976|              Q2|
    | a4|GINGER| KN231SD|       1975|              Q1|
    | a5|  POSH| HK192YC|       1973|              Q3|
    +---+------+--------+-----------+----------------+

    """

    df = df.withColumn(
        date_column,
        col(date_column).cast(
            StringType()
        )
    )

    df = df.withColumn(date_column,
                       regexp_replace(
                        col(date_column),
                       '[^0-9]',
                       ""))

    df = df.withColumn(
        f'{date_column}_year',
        col(date_column).substr(
            1,4)
    )

    df = df.withColumn(
        f'{date_column}_year_quarter',
        col(date_column).substr(
            5,2)
    )

    replace_dict = {'01': 'Q1',
                    '02': 'Q1',
                    '03': 'Q1',
                    '04': 'Q2',
                    '05': 'Q2',
                    '06': 'Q2',
                    '07': 'Q3',
                    '08': 'Q3',
                    '09': 'Q3',
                    '10': 'Q4',
                    '11': 'Q4',
                    '12': 'Q4',}

    df = df.na.replace(replace_dict,
                       subset=f'{date_column}_year_quarter')

    for i in [f'{date_column}_year', f'{date_column}_year_quarter']:
        df = df.withColumn(i,
                           when(~col(f'{date_column}_year_quarter').isin(['Q1','Q2','Q3','Q4']),
                                'input_error')
                           .when(length(col(date_column)) != 8,
                                 'input_error')
                           .otherwise(col(i)))

    df = df.drop(date_column)

    return df


###############################################################################

def check_variables(df, approved_vars):

    """
    Checks that all variable names are present in the database for export but also 
    that the database does not have any extra variables not approved by the stakeholders.
  
    This function returns a python dataframe of which approved variables 
    are not in the database for export and another dataframe of any variables 
    in the database that shouldn't be there.
  
    Parameters
    ----------
    df: dataframe
      Dataframe to which the function is applied.
    approved_vars: a list of variables in string format

    Returns
    -------
    The function returns a tuple of two python dataframes:
    a
      The variable names that have been approved by the stakeholders 
      that are not in the database for export.
    x
      The variable names that are in the database for export but have 
      not been approved by the stakeholders.
  
    Example
    -------

    > df.show()
    +---+--------+------+--------+
    | id|     dob|  name|postcode|
    +---+--------+------+--------+
    | a1|19990121|SPORTY| PO356TH|
    | a2|19771109| SCARY|   E34TO|
    | a3|19760424|  BABY| BH242ED|
    | a4|19251217|GINGER| KN231SD|
    | a5|19730724|  POSH| HK192YC|
    +---+--------+------+--------+

    As the function returns a tuple, call in the function using tuple unpacking

    approved = ['approved1', 'approved2', 'id', 'name', 'postcode']

    > missing,not_approved = hash_id(df,approved_vars = approved) 

    > missing  
  	  in_approved_not_in_df
    0	approved1
    1	approved2

    > not_approved
  	  in_df_not_in_approved
    0	dob

    """

    a_list = [a.lower() for a in approved_vars]

    df_list = [c.lower() for c in df.columns]

    a = pd.DataFrame({'in_approved_not_in_df'
                      :
                      [a for a in a_list if a not in df_list]})

    x = pd.DataFrame({'in_df_not_in_approved'
                      :
                      [x for x in df_list if x not in a_list]})

    return a,x

###--- END

###############################################################################


def numerical_outliers(df, numeric_column):
    """
    Provides a dataframe of outliers on your chosen column.

    prerequisites
    -------------
    The dataframe needs to be a Pyspark database.
    The column to calculate outliers must have only numeric characters and a maximum
    of one decimal point. However, the numeric_column type can be a string, integer
    float, long or bigint as the function will turn it all to a float.
    Make sure numeric_column is not truncated.

      Parameters
    ----------
    df: dataframe
      Pyspark dataframe to which the function is applied.
    numeric_column: column that the outliers will be calculated on (str)
      The name of the column you want to calculate your outliers on.

    Returns
    -------
    It will return the original dataframe filtered to only include records that are
    outliers on the desired variable.


    Example
    -------

    > df_nums.sample(False, 0.1, seed=0).limit(5).show(5)
    +------+----+
    |number|  id|
    +------+----+
    |   451|id21|
    |   411|id34|
    |   674|id40|
    |   697|id45|
    |   412|id49|
    +------+----+

    > number_outliers = numerical_outliers(df=df_nums, numeric_column='number')
    .
    number_outliers.show()
    +------+-----+-------------------+
    |number|   id|            Outlier|
    +------+-----+-------------------+
    |    17|id179| lower_bound = 90.0|
    |    46|id180| lower_bound = 90.0|
    |    41|id181| lower_bound = 90.0|
    |   984|id176|upper_bound = 908.0|
    |   977|id177|upper_bound = 908.0|
    |   965|id178|upper_bound = 908.0|
    +------+-----+-------------------+

    """

    num_df = df.select(numeric_column).withColumn('w', lit('w'))

    num_df = num_df.withColumn(
        numeric_column,
        col(numeric_column).cast(
            FloatType()
        )
    )

    num_df.registerTempTable("num_df")

    quartile_1 = spark.sql('SELECT DISTINCT ' + numeric_column
                           + ', PERCENTILE(' + numeric_column + ',0.25) OVER (PARTITION BY w) \
                           from num_df').collect()[1][1]

    quartile_3 = spark.sql('SELECT DISTINCT ' + numeric_column
                           + ', PERCENTILE(' + numeric_column + ',0.75) OVER (PARTITION BY w) \
                           from num_df').collect()[1][1]

    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)

    df_lower = df.filter(col(numeric_column)
                         < lower_bound).withColumn('Outlier',
                                                   concat(lit('lower_bound = '),lit(lower_bound)))

    df_upper = df.filter(col(numeric_column)
                         > upper_bound).withColumn('Outlier',
                                                   concat(lit('upper_bound = '),lit(upper_bound)))

    df = df_lower.union(df_upper)

    return df

###############################################################################


def numerical_percentiles(df, numeric_column, greater_or_less, percentile):
    """
    Provides the original database filtered to include only records where
    the value on the numeric_column is greater or less than the percentile value.

    prerequisites
    -------------
    The dataframe needs to be a Pyspark database.
    Numeric_column must have only numeric characters and a maximum
    of one decimal point. However, the numeric_column type can be a string,
    integer, float, long or bigint as the function will turn it all to a float.
    Make sure numeric_column is not truncated.
    The percentile value needs to be assigned in a fraction as a string.

      Parameters
    ----------
    df: dataframe
      Pyspark dataframe to which the function is applied.
    numeric_column: column that the outliers will be calculated on (str)
      The name of the column you want to calculate the percentile on.
    greater_or_less: Two choices: 'greater' or 'less' (str)
      Assign the string 'greater' or 'less' if you want the output to include
      records that are either greater or less than the percentile value.
    prcentile: The value you want calculated (str)
      Assign this value as a fraction and a string.

    Returns
    -------
    It will return the original dataframe filted to only include records that are
    outliers on the desired variable.


    Example
    -------

    > df_nums.sample(False, 0.1, seed=0).limit(5).show(5)
    +------+----+
    |number|  id|
    +------+----+
    |   451|id21|
    |   411|id34|
    |   674|id40|
    |   697|id45|
    |   412|id49|
    +------+----+

    > percentile = numerical_percentiles(df=df_nums,
                                         numeric_column='number',
                                         greater_or_less = 'less',
                                         percentile = '0.15')

    > percentile.show(5)
    +------+----+---------------------+
    |number|  id|0.15_percentile_value|
    +------+----+---------------------+
    |   353| id6|               357.25|
    |   348|id11|               357.25|
    |   322|id24|               357.25|
    |   301|id27|               357.25|
    |   307|id28|               357.25|
    +------+----+---------------------+

    """
    num_df = df.select(numeric_column).withColumn('w',lit('w'))

    num_df = num_df.withColumn(
        numeric_column,
        col(numeric_column).cast(
            FloatType()
        )
    )

    num_df.registerTempTable("num_df")

    if greater_or_less == 'less':

        val_percent = spark.sql('SELECT DISTINCT ' + numeric_column
                                + ', PERCENTILE(' + numeric_column + ',' + percentile
                                + ') OVER (PARTITION BY w) from num_df').collect()[1][1]

        df_percent = df.filter(col(numeric_column)
                               < val_percent).withColumn(percentile + '_percentile_value',
                                                         lit(val_percent))

    elif greater_or_less == 'greater':

        val_percent = spark.sql('SELECT DISTINCT ' + numeric_column
                                + ', PERCENTILE(' + numeric_column + ',' + percentile
                                + ') OVER (PARTITION BY w) from num_df').collect()[1][1]

        df_percent = df.filter(col(numeric_column)
                               > val_percent).withColumn(percentile + '_percentile_value',
                                                         lit(val_percent))
    else:
        None

    return df_percent


###############################################################################

def postcode_level(df, column, postcode_level):

    """
    Deletes your postcode column from your dataframe and replaces it with your
    chosen level of postcode. 
  
    Parameters
    ----------
    df: dataframe
      Dataframe to which the function is applied.It must be a pyspark dataframe.
    column: postcode column (str)
      The name of the postcode column to change to a different level of geography.
      Spaces and punctuation can be present and it is not case sensitive as the function 
      has a work around for this.
    postcode_level: choose either 'area', 'district' or 'sector'.
      Type the postcode level as a string.

    Returns
    -------
    The function will return the original dataframe with the postcode column
    replaced with its respective chosen georgraphy level.

    Example
    -------

    > df.show()
    +---+-----------+------+--------+------------+
    | id|        dob|  name|postcode|        fans|
    +---+-----------+------+--------+------------+
    | a1|QQ 123456 C|SPORTY| PO356TH|           1|
    | a2|07451224152| SCARY|KZ66 7YT|     1831687|
    | a3|   19760424|  BABY|  B242ED|     1932435|
    | a4|   19750127|GINGER| KN231SD|      969537|
    | a5|   19730724|  POSH| HK192YC|987646314834|
    +---+-----------+------+--------+------------+
        
    > for i in ['area', 'district','sector']:
        postcode_level(df=df,
                       column='postcode',
                       postcode_level=i).show(5) 
  
    Output

    +---+-----------+------+------------+-------------+
    | id|        dob|  name|        fans|postcode_area|
    +---+-----------+------+------------+-------------+
    | a1|QQ 123456 C|SPORTY|           1|           PO|
    | a2|07451224152| SCARY|     1831687|           KZ|
    | a3|   19760424|  BABY|     1932435|            B|
    | a4|   19750127|GINGER|      969537|           KN|
    | a5|   19730724|  POSH|987646314834|           HK|
    +---+-----------+------+------------+-------------+

    +---+-----------+------+------------+-----------------+
    | id|        dob|  name|        fans|postcode_district|
    +---+-----------+------+------------+-----------------+
    | a1|QQ 123456 C|SPORTY|           1|             PO35|
    | a2|07451224152| SCARY|     1831687|             KZ66|
    | a3|   19760424|  BABY|     1932435|              B24|
    | a4|   19750127|GINGER|      969537|             KN23|
    | a5|   19730724|  POSH|987646314834|             HK19|
    +---+-----------+------+------------+-----------------+

    +---+-----------+------+------------+---------------+
    | id|        dob|  name|        fans|postcode_sector|
    +---+-----------+------+------------+---------------+
    | a1|QQ 123456 C|SPORTY|           1|          PO356|
    | a2|07451224152| SCARY|     1831687|          KZ667|
    | a3|   19760424|  BABY|     1932435|           B242|
    | a4|   19750127|GINGER|      969537|          KN231|
    | a5|   19730724|  POSH|987646314834|          HK192|
    +---+-----------+------+------------+---------------+
    """

    df = df.withColumn(column,
                       regexp_replace(
                       col(column),
                       '[^a-zA-Z0-9]',
                       ""))

    if postcode_level == 'area':
        df = df.withColumn('postcode_'+postcode_level,
                           regexp_extract(column,
                                          '^(?:(?![0-9]).)*',
                                          idx = 0))

    elif postcode_level == 'district':
        df = df.withColumn('postcode_'+postcode_level,
                           expr(f'substring({column}, 1, length({column})-3)'))

    elif postcode_level == 'sector':
        df = df.withColumn('postcode_'+postcode_level,
                           expr(f'substring({column}, 1, length({column})-2)'))

    else:
        None

    df = df.drop(column)

    return df

###############################################################################

def identifying_strings(df, required_identifiers):
    """
    You can choose which regex expressions you want to search for in the 
    regex database from the data folder.
  
    Returns your dataframes with a corresponding identifying column for 
    each column in the dataframe. In the corresponding identifying column, 
    it will say if a regex expression was found and which regex expression 
    was used to find it.
  
    It also provides a database of the regex expressions you have chosen 
    for you to document as the regex expressions may change over time.
  
    Parameters
    ----------
    df: dataframe
      Dataframe to which the function is applied.
    required_identifiers: regex expressions you want to search for
      input the regex expressions that you want to search for as a 
      list of strings
  
    
    Returns
    -------
    The function returns a tuple of two dataframes:
      found identifying strings
        Returns your dataframes with a corresponding identifying column for 
        each column in the dataframe. In the corresponding identifying column, 
        it will say if a regex expression was found and which regex expression 
        was used to find it.
      regex database
        A database of the regex expressions you have used in this function 
        for you to document
    Example
    -------
  
    > df.show()
    +-----------+-----------+--------+----------+
    |         id|        dob|    name|  postcode|
    +-----------+-----------+--------+----------+
    |         a1|QQ 123456 C|  SPORTY|   PO356TH|
    |         a2|   19830914|   SCARY|  KZ66 ZYT|
    |         a3|   19760424|    BABY|    B242ED|
    |         a4|   19750127|  GINGER|   KN231SD|
    |         a5|   19730724|    POSH|   HK192YC|
    |ABE46372837|   19631011| SKINNER|    SP63HW|
    |         a7|   19780306|FLANDERS|6878844321|
    |         a8|07451224152|  BARNEY|  PO29 2YC|
    |         a9|   19630507|  KRUSTY|    NP65LN|
    |        a10|   19981218|MILHOUSE|  MK17 6PO|
    +-----------+-----------+--------+----------+
  
    > search_for = ['NIN', 
                'UK Vehicle Reg',
                'UK phone no',
                'NHS No',
                'UK Child Benefit']
    > identified, regex_used = identifying_strings(df=df, 
                      required_identifiers=search_for)
                    
    > identified.show()
    +-----------+--------+----------------+--------------+---------------+-------------------+
    |         id|    name|   id_identifier|dob_identifier|name_identifier|postcode_identifier|
    +-----------+--------+----------------+--------------+---------------+-------------------+
    |         a8|  BARNEY|            null|   UK phone no|           null|               null|
    |         a7|FLANDERS|            null|          null|           null|             NHS No|
    |         a2|   SCARY|            null|          null|           null|     UK Vehicle Reg|
    |         a1|  SPORTY|            null|           NIN|           null|               null|
    |ABE46372837| SKINNER|UK Child Benefit|          null|           null|               null|
    +-----------+--------+----------------+--------------+---------------+-------------------+
  
    > spark.createDataFrame(regex_used).show()
    +----------------+--------------------+--------------+-----+
    |      Identifier|               Regex|        Origin|Notes|
    +----------------+--------------------+--------------+-----+
    |          NHS No|            ^\d{10}$|      de_utils|  NaN|
    |             NIN|^\s*[a-zA-Z]{2}(?...|Stack Overflow|  NaN|
    |     UK phone no|^(?:0|\+?44)(?:\d...|  RegExLib.com|  NaN|
    |UK Child Benefit|     ^[A-Z]{3}\d{8}$|  RegExLib.com|  NaN|
    |  UK Vehicle Reg|^([A-Z]{3}\s?(\d{...|  RegExLib.com|  NaN|
    +----------------+--------------------+--------------+-----+
  
    """

    identifier_dataset = pd.DataFrame(regex_data, columns=regex_cols)

    regex_dict = dict(zip(identifier_dataset.Identifier,
                        identifier_dataset.Regex))

    required_dict = {key: regex_dict[key] for key in regex_dict.keys()
                          & {i for i in required_identifiers}}

    required_regex = list(required_dict.values())

    new_df = spark.createDataFrame([], schema=df.schema)

    for i in df.columns:
        for l in required_regex:
            filtered = df.filter(col(i).rlike(l))
            new_df = new_df.union(filtered)

    for n in new_df.columns:
        new_df = new_df.withColumn(n+'_identifier', lit(None))
        for k in required_dict:
            new_df = new_df.withColumn(n+'_identifier',
                                       when(col(n).rlike(required_dict.get(k)), k)
                                       .otherwise(col(n+'_identifier')))

    new_df = new_df.distinct()

    regex_used = identifier_dataset[identifier_dataset['Identifier']
                                    .isin(required_identifiers)]

    return new_df, regex_used
