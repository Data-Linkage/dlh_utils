"""
Proof of concept script for use of check_sum in IDA QA proceedure.

Check_sum is a consistent numerical hash that can be used to represent a sequence of 
row-wise values. If applied to common variables, it can therefore be used to
check that values in a row are consistent between raw and indexed versions of data. 

Script contains functions for:

- generating mock data
- generating a check_sum variable
- check sum QA (comparison between 2 dataframes)
- A generalised multiple types fillna function (needed for reducing false positives 
in check_sum_qa, as explained in the script)

There are also mock data demonstrations of the check_sum_qa and fillna functions 

"""


from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.types import *
import pandas as pd
from string import ascii_lowercase 
from random import randint, sample

spark = SparkSession\
    .builder\
    .appName("dev")\
    .getOrCreate()
    
###############################################################################
def random_string(min_l=2,max_l=10):
  
  '''Creates random lowercase string between specified minimum and maximum 
  length. Used in creating test dataframe. Addittionally creates nulls in 10%
  of values
  
  min_l - minimum string length
  max_l - maximum string length
  '''
  
  # chance treturn null
  if (randint(0,9))==0:
    return None
  # else return random string
  else:
    return ''.join(sample(ascii_lowercase,randint(min_l,max_l)))
  

###############################################################################
def fill_null_multiple_types(
  df,
  string_fill = '',
  boolean_fill = False,
  numeric_fill = 0,
):

  '''Allows fill of nulls in multiple variable types (i.e. where as an integer
  column null filled with '' would be NA and a string column null filled with 0
  would remain null). This is used in check_sum_qa to reduce chance of false 
  positives (i.e. whereas hash of a|b|null and a|null|b would be the same value
  a hashes of a|b|'' and a|''|b would be distinct) - see demo below.
  
  If any of the fill arguments are None, these will be skipped
  
  df - target dataframe
  string_fill - value fill for string type columns
  boolean_fill - value fill for boolean type columns
  numeric fill - value fill for numeric columns (derived as those with type not
  equal to string, boolean or null).
  
  NOTE: null type / entierly null columns cannot be filled using fillna. However 
  the position of these nulls in a row wise sequence will be consistent and therefore
  will not affect the check sum

  '''  
  
  types = df.dtypes
  string_cols = [x[0] for x in types if x[1]=='string']
  booleans_cols = [x[0] for x in types if x[1]=='boolean']
  null_cols = [x[0] for x in types if x[1]=='null']
  # cols that are not string, boolean, or null
  numeric_cols = [x for x in df.columns 
                  if x not in string_cols+booleans_cols+null_cols]

  if string_fill is not None:
    df = df.fillna({col:string_fill for col in string_cols})

  if boolean_fill is not None:
    df = df.fillna({col:boolean_fill for col in booleans_cols})

  if numeric_fill is not None:
    df = df.fillna({col:numeric_fill for col in numeric_cols})
    
  return df


###############################################################################

def check_sum(df,out_col='check_sum',check_sum_only=True):
  
  """Creates total row-wise check_sum variable (for illustration)
  
  df - target dataframe
  out_col - output check sum column name
  check_sum_only - If true, returns single check_sum column dataframe, else
  returns original dataframe with addittional check_sum column
  """
  
  # creates temporary virtual SQL table allowing SQL to be executed
  df.createOrReplaceTempView('check_sum_temp')
  
  if check_sum_only:
    return spark.sql(f"SELECT HASH(*) AS {out_col} FROM check_sum_temp")
  else:
    return spark.sql(f"SELECT *,HASH(*) AS {out_col} FROM check_sum_temp")
    

###############################################################################
def check_sum_qa(df_1,df_2):
  
  """Enables consistency check of common columns in two dataframes using check 
  sum
  """
  
  # identifies common columns
  common_cols = [col for col in df_1.columns 
                 if col in df_2.columns]
  
  # reduces dtaframes to common columns within function 
  df_1 = df_1.select(common_cols)
  df_2 = df_2.select(common_cols)
  
  # consistently fills nulls in both dtaframes
  # this is to reduce possibility of false positives
  df_1 = fill_null_multiple_types(
    df_1,
    string_fill = '',
    boolean_fill = True,
    numeric_fill = 0,
  )
  df_2 = fill_null_multiple_types(
    df_2,
    string_fill = '',
    boolean_fill = True,
    numeric_fill = 0,
  )
  
  # creates temporary tables views in order to apply SQL
  df_1.createOrReplaceTempView('df_1_temp')
  df_2.createOrReplaceTempView('df_2_temp')
  
  # returns single column of row check sums for each dataframe
  # persisted for multiple joins below
  df_1 = spark.sql("SELECT HASH(*) AS check_sum FROM df_1_temp").persist()  
  df_2 = spark.sql("SELECT HASH(*) AS check_sum FROM df_2_temp").persist()
  
  # Applies left anti join in both directions
  # this is to identify if any check sum values are inconsistent between dataframes
  # the max function returns the maximum number of inconsitencies for the two joins
  # this is to detect any inconsitencies in either direction
  inconsistency_count = max([
    (df_1
     .join(df_2,
           on = 'check_sum',
           how = 'left_anti'
          )
    ).count(),
    (df_2
     .join(df_1,
           on = 'check_sum',
           how = 'left_anti'
          )
    ).count(),
                            ])
  
  # check sum dataframes unpersisted after inconsistency_count is determined
  df_1.unpersist()
  df_2.unpersist()
  
  # returns true or false qa check depending on inconsitencies detected
  if inconsistency_count == 0:
    return True
  else:
    return False
    
###############################################################################
###############################################################################
""" Proof of Concept demonstration - check_sum_qa
"""

# creates dataframe of random strings n_cols x n_rows
# tested up to 1000 variables and 100 rows
# will require scalability testing in DAP
n_cols = 10
n_rows = 10

df = (spark.createDataFrame(
  pd.DataFrame(
    {col:[random_string() for row in range(n_rows)] 
     for col in range(n_cols)}
  )
)).persist()

df.count()
# can be toggled to view df if required
#df.show()
#check_sum(df,out_col='check_sum',check_sum_only=False).show()

# runs check_sum qa for identical dataframes
# this will return True
print(check_sum_qa(df,df))

# Creates discrepency between df and df_2
df_2 = df.select(df.columns)
df_2 = (df_2
       .withColumn('0',F.lit(None))
       )

# runs check_sum qa for dataframes with discrepency
# this will return False
print(check_sum_qa(df,df_2))

# unpersists demonstration dataframe
df.unpersist()

###############################################################################
###############################################################################
""" Proof of Concept demonstration - fill_null_multiple_types
"""

df = (spark.createDataFrame(
  pd.DataFrame(
    {
      'string':['a','b','c',None],
      'numeric':[1,2,3,None],
      'boolean':[False,True,False,None],
    }
  )
))

df.show()

# illustrating how type of fill is only applicable to same type variables
df.fillna('').show(4,False)
df.fillna(0).show(4,False)
df.fillna(False).show(4,False)
# enables fill of multiple types
fill_null_multiple_types(df).show(4,False)

# illustrating need to fill nulls
# will return false positive with out of sequence null values 
spark.sql("SELECT HASH(NULL, 'a', 'b'), HASH('a', NULL , 'b')").show(1,False)
# will return distinct check_sum values with nulls filled
spark.sql("SELECT HASH('', 'a', 'b'), HASH('a', '' , 'b')").show(1,False)
