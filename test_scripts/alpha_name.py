#test script to test linkage.alphaname


import pyspark.sql.functions as F
#import to_timestamp, col, datediff, months_between
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

spark = (
    SparkSession.builder.appName("dataframe_testing")
    .config("spark.executor.memory", "5g")
    .config("spark.yarn.excecutor.memoryOverhead", "2g")
    .getOrCreate()
)



##########################
#PASTE THIS BEFORE IMPORTS ON MODULE RESR
#proritise local installation over dlh_dev installed module, to fix console execution
recovery = sys.path
sys.path.insert(0,f'{os.getcwd()}/dlh_utils')
##########################




###########function to later go into linkage


###############################################################################


def alpha_name(df, input_col, output_col):
    """
    Orders each field of a string column alphabetically, also setting to UPPER CASE.

    Parameters
    ----------
    df: Spark dataframe
    input_col: string
      name of column to be sorted alphabetically
    output_col: string
      name of column to be output

    Returns
    -------
    a dataframe with output_col appended
    null values in input_col persist to output_col

    Raises
    ------
    Exception if input_col not string type
    
    Example
    --------

    > df.show()
    +---+--------+
    | ID|Forename|
    +---+--------+
    |  1|   Homer|
    |  2|   Marge|
    |  3|    Bart|
    |  4|    Lisa|
    |  5|  Maggie|
    |  6|    null|
    +---+--------+

    > alpha_name(df,'Forename','alphaname').show()
    +---+--------+---------+
    | ID|Forename|alphaname|
    +---+--------+---------+
    |  1|   Homer|    EHMOR|
    |  2|   Marge|    AEGMR|
    |  3|    Bart|     ABRT|
    |  4|    Lisa|     AILS|
    |  5|  Maggie|   AEGGIM|
    |  6|    null|     null|
    +---+--------+---------+

    """
    
    #input validation
    if (df.schema[input_col].dataType.typeName()!='string'):
      raise Exception(f'Column: {input_col} is not of type string')
    
    else:
      #concat removes any null values. conditional replacement only when not null added
      #to avoid unwanted removal of null
      df= df.withColumn(output_col, \
                F.when(F.col(input_col).isNull(),F.col(input_col)).otherwise(\
                F.concat_ws('',F.array_sort(F.split(F.upper(F.col(input_col)),'')))))

    return df




###############test data to later go into test_linkage

test_schema1 = StructType([
  StructField("ID", IntegerType(), True),
  StructField("Forename", StringType(), True),  
])
test_data1 = [
  [1, "Homer"],
  [2, "Marge"],
  [3, "Bart"],
  [4, "Lisa"],
  [5, "Maggie"],
  [6, None],
]

test_df1 = spark.createDataFrame(test_data1, test_schema1)






alpha_name(test_df1,'Forename','alphaname').show()






test_schema2 = StructType([
  StructField("ID", IntegerType(), True),
  StructField("Forename", StringType(), True),  
])
test_data2 = [
  [1, "Romer, Bogdan"],
  [2, "Margerine"],
  [3, None],
  [4, "Nisa"],
  [5, "Moggie"],
]

test_df2 = spark.createDataFrame(test_data2, test_schema2)

alpha_name(test_df2,'Forename','alphaname').show()








