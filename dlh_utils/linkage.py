'''
Functions used within the linkage phase of data linkage projects
'''
import os
import pandas as pd
import jellyfish
from copy import deepcopy
import re
import py4j
from pyspark.sql import SparkSession, Window
import pyspark.sql.functions as F
from pyspark.sql.types import StringType, FloatType
from graphframes import *
from dlh_utils import dataframes as da
from dlh_utils import utilities as ut

# phonetic encoders


def alpha_name(df, input_col, output_col):
    """
    Orders string columns alphabetically, also setting them to UPPER CASE.

    Parameters
    ----------
    df: dataframe
    input_col: string
      name of column to be sorted alphabetically
    output_col: string
      name of column to be output

    Returns
    -------
    a dataframe with output_col appended

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
    +---+--------+---------+

    """
    df = df.withColumn('name_array', (F.split(F.upper(F.col(input_col)), '')))\
           .withColumn('sorted_name_array', F.array_sort(F.col('name_array')))\
           .withColumn(output_col, F.concat_ws('', F.col('sorted_name_array')))\
           .drop('name_array', 'sorted_name_array') 
    return df

###############################################################################


def metaphone(df, input_col, output_col):
    """
    Generates the metaphone phonetic encoding of a string.

    Parameters
    ----------
    df: dataframe
    input_col: string
      name of column to create metaphone encoding on
    output_col: string
      name of column to be output

    Returns
    -------
    A df with output_col appended

    Example
    --------

  > metaphone(df,'Forename','metaname').show()
    +---+---------+------------------+
    | ID| Forename|forename_metaphone|
    +---+---------+------------------+
    |  1|    David|               TFT|
    |  2|  Idrissa|              ITRS|
    |  3|   Edward|             ETWRT|
    |  4|   Gordon|              KRTN|
    |  5|     Emma|                EM|
    +---+---------+------------------+

    """
    @F.udf(returnType=StringType())
    def meta(s):
        return None if s == None else jellyfish.metaphone(s)

    df = df.withColumn(output_col, meta(F.col(input_col)))

    return df

###############################################################################


def soundex(df, input_col, output_col):
    """
    Generates the soundex phonetic encoding of a string.

    Parameters
    ----------
    df: dataframe
    input_col: string
      name of column to create soundex encoding on
    output_col: string
      name of column to be output

    Returns
    -------
    A df with output_col appended

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
    +---+--------+

    > soundex(df,'Forename','forename_soundex').show()
    +---+--------+----------------+
    | ID|Forename|forename_soundex|
    +---+--------+----------------+
    |  1|   Homer|            H560|
    |  2|   Marge|            M620|
    |  3|    Bart|            B630|
    |  4|    Lisa|            L200|
    |  5|  Maggie|            M200|
    +---+--------+----------------+

    """
    df = df.withColumn(output_col, F.soundex(input_col))
    return df

###############################################################################
# string comparators


def std_lev_score(string1, string2):
    """
    Applies the standardised levenshtein string similarity function to two 
    strings to return a score between 0 and 1.

    This function works at the column level, and so needs to either be applied
    to two forename columns in an already-linked dataset, or as a join
    condition in a matchkey. See example for both scenarios outlined.

    Parameters
    ----------
    string1: str
        string to be compared to string2
    string2: str
        string to be compared to string1

    Returns
    -------
    float
        similarity score between 0 and 1

    Example
    --------

    for a pre-joined dataset:

    > df.show()
    +---+--------+----------+
    | ID|Forename|Forename_2|
    +---+--------+----------+
    |  1|   Homer|  Milhouse|
    |  2|   Marge|  Milhouse|
    |  3|    Bart|  Milhouse|
    |  4|    Lisa|  Milhouse|
    |  5|  Maggie|  Milhouse|
    +---+--------+----------+

    > df = df.withColumn('forename_lev', std_lev_score(F.col('Forename'), F.col('Forename_2')))
    +---+--------+----------+------------+
    | ID|Forename|Forename_2|forename_lev|
    +---+--------+----------+------------+
    |  1|   Homer|  Milhouse|       0.125|
    |  2|   Marge|  Milhouse|        0.25|
    |  3|    Bart|  Milhouse|         0.0|
    |  4|    Lisa|  Milhouse|        0.25|
    |  5|  Maggie|  Milhouse|        0.25|
    +---+--------+----------+------------+

    Example 2
    ---------

    used in a matchkey:

    MK = [linkage.std_lev_score(F.col('First_Name_cen'), F.col('First_Name_ccs')) > 0.7,
        CEN.Last_Name_cen == CCS.Last_Name_ccs,
        CEN.Sex_cen == CCS.Sex_ccs,
        CEN.Resident_Age_cen == CCS.Resident_Age_ccs,
        CEN.Postcode_cen == CCS.Postcode_ccs]

    links = linkage.deterministic_linkage(df_l = CEN, df_r = CCS, id_l = 'Resident_ID_cen', id_r = 'Resident_ID_ccs',
                               matchkeys = MK, out_dir = '/some_path/links')
    """

    return (1 - ((F.levenshtein(string1, string2)) /
                 F.greatest(F.length(string1), F.length(string2))))

###############################################################################


@F.udf(FloatType())
def jaro(string1, string2):
    """
    Applies the Jaro string similarity function to two strings and calculates
    a score between 0 and 1.

    This function works at the column level, and so needs to either be applied
    to two forename columns in an already-linked dataset, or as a join
    condition in a matchkey. See example for both scenarios outlined.

    Parameters
    ----------
    string1: str
        string to be compared to string2
    string2: str
        string to be compared to string1

    Returns
    -------
    float
        similarity score between 0 and 1

    Example
    --------

    for a pre-joined dataset:
    
    >df.show()    
    +---+--------+----------+
    | ID|Forename|Forename_2|
    +---+--------+----------+
    |  1|   Homer|      John|
    |  2|   Marge|      John|
    |  3|    Bart|      John|
    |  4|    Lisa|      John|
    |  5|  Maggie|      John|
    +---+--------+----------+

    >df = df.withColumn('Forename_jaro', jaro(F.col('Forename'), F.col('Forename_2')))
    >df.show()
    +---+--------+----------+-------------+
    | ID|Forename|Forename_2|Forename_jaro|
    +---+--------+----------+-------------+
    |  1|   Homer|      John|   0.48333332|
    |  2|   Marge|      John|          0.0|
    |  3|    Bart|      John|          0.0|
    |  4|    Lisa|      John|          0.0|
    |  5|  Maggie|      John|          0.0|
    +---+--------+----------+-------+-----+

    Example 2
    ---------

    used in a matchkey:

    MK = [linkage.jaro(F.col('First_Name_cen'), F.col('First_Name_ccs')) > 0.7,
        CEN.Last_Name_cen == CCS.Last_Name_ccs,
        CEN.Sex_cen == CCS.Sex_ccs,
        CEN.Resident_Age_cen == CCS.Resident_Age_ccs,
        CEN.Postcode_cen == CCS.Postcode_ccs]

    links = linkage.deterministic_linkage(df_l = CEN, df_r = CCS, id_l = 'Resident_ID_cen', id_r = 'Resident_ID_ccs',
                               matchkeys = MK, out_dir = '/some_path/links')
    """

    return jellyfish.jaro_similarity(
        string1, string2) if string1 is not None and string2 is not None else None

###############################################################################


@F.udf(FloatType())
def jaro_winkler(string1, string2):
    """
    Applies the Jaro Winkler string similarity function to two strings and
    calculates a score between 0 and 1.

    This function works at the column level, and so needs to either be applied
    to two forename columns in an already-linked dataset, or as a join
    condition in a matchkey. See example for both scenarios outlined.

    Parameters
    ----------
    string1: str
        string to be compared to string2
    string2: str
        string to be compared to string1

    Returns
    -------
    float
        similarity score between 0 and 1

    Example
    --------

    >df.show()
    +---+---------+----------+
    | ID| Forename|Forename_2|
    +---+---------+----------+
    |  1|    David|     Emily|
    |  2|  Idrissa|     Emily|
    |  3|   Edward|     Emily|
    |  4|   Gordon|     Emily|
    |  5|     Emma|     Emily|
    +---+---------+----------+

    >df = df.withColumn('fnjaro_winkler', jaro_winkler(F.col('Forename'), F.col('Forename_2')))
    +---+---------+----------+--------------+
    | ID| Forename|Forename_2|fnjaro_winkler|
    +---+---------+----------+--------------+
    |  1|    David|     Emily|    0.46666667|
    |  2|  Idrissa|     Emily|    0.44761905|
    |  3|   Edward|     Emily|    0.45555556|
    |  4|   Gordon|     Emily|           0.0|
    |  5|     Emma|     Emily|     0.6333333|
    +---+---------+----------+--------------+

    Example 2
    ---------

    MK = [linkage.jaro_winkler(F.col('First_Name_cen'), F.col('First_Name_ccs')) > 0.7,
        CEN.Last_Name_cen == CCS.Last_Name_ccs,
        CEN.Sex_cen == CCS.Sex_ccs,
        CEN.Resident_Age_cen == CCS.Resident_Age_ccs,
        CEN.Postcode_cen == CCS.Postcode_ccs]

    links = linkage.deterministic_linkage(df_l = CEN, df_r = CCS, id_l = 'Resident_ID_cen', id_r = 'Resident_ID_ccs',
                               matchkeys = MK, out_dir = '/user/username/cen_ccs_links')

    """

    return jellyfish.jaro_winkler_similarity(
        string1, string2) if string1 is not None and string2 is not None else None

###############################################################################
# linkage methods


def blocking(df1, df2, blocks, id_vars):
    """ 
    Combines two spark dataframes, based on a set of defined blocking criteria,
    to create a new dataframe of unique record pairs.

    Parameters
    ----------
    df1: DataFrame
    df2: DataFrame
    blocks: Dictionary
        pairs of variables from df1 and df2 to block on, each item is a new
        blocking pass.
    id_vars: List
        unique ID variables from df1 and df2

    Returns
    -------
    combined_blocks
      a new dataframe containing unique pairs of blocked records

    Example
    ------
    > id_vars = ['ID_1', 'ID_2']

    > blocks = {'pc_df1': 'pc_df2'}

    > df1.show()
    +----+-------+-------+------+
    |ID_1|age_df1|sex_df1|pc_df1|
    +----+-------+-------+------+
    |   1|      1|   Male|gu1111|
    |   2|      1| Female|gu1211|
    |   3|     56|   Male|gu2111|
    +----+-------+-------+------+

    > df2.show()
    +----+-------+-------+------+
    |ID_2|age_df2|sex_df2|pc_df2|
    +----+-------+-------+------+
    |   6|      2| Female|gu1211|
    |   5|     56|   Male|gu1411|
    |   4|      7| Female|gu1111|
    +----+-------+-------+------+

    > blocking(df1, df2, blocks, id_vars).show()
    +----+-------+-------+------+----+-------+-------+------+
    |ID_1|age_df1|sex_df1|pc_df1|ID_2|age_df2|sex_df2|pc_df2|
    +----+-------+-------+------+----+-------+-------+------+
    |   1|      1|   Male|gu1111|   4|      7| Female|gu1111|
    |   2|      1| Female|gu1211|   6|      2| Female|gu1211|
    |   3|     56|   Male|gu2111|   5|     56|   Male|gu2111|
    +----+-------+-------+------+----+-------+-------+------+
    """

    for index, (key, value) in enumerate(blocks.items(), 1):

        if index == 1:
            first_block = df1.join(df2, df1[key] == df2[value], how='inner')
            print(f"block {index} contains", str(
                first_block.count()), "records")

        if len(blocks.items()) == 1:
            combined_blocks = first_block

        elif index > 1:
            combined_blocks = df1.join(
                df2, df1[key] == df2[value], how='inner')
            print(f"block {index} contains", str(
                combined_blocks.count()), "records")
            combined_blocks = combined_blocks.union(
                first_block).drop_duplicates(id_vars)

    return combined_blocks

###############################################################################


def cluster_number(df, id_1, id_2):
    """ 
    Takes dataframe of matches with two id columns (id_1 and id_2) and assigns
    a cluster number to the dataframe based on the unique id pairings.

    PLEASE NOTE: this function relies on a sparksession that has been initiated with
    external graphframes JAR dependencies added to the session. Without this, you may
    encounter this exception when calling the function:

    java.lang.ClassNotFoundException: org.graphframes.GraphFramePythonAPI

    This can either be fixed by starting a sparksession from the `sessions` module of
    this package, or by adding the JAR files from within the graphframes-wrapper 
    package to your spark session, by setting the "spark.jars" spark config parameter
    equal to the path to these JAR files.

    Parameters
    ----------
    df: DataFrame
        DataFrame to add new column 'Cluster_Number' to.
    id_1: string
        ID column of first DataFrame.
    id_2: string
        ID column of second DataFrame.

    Raises
    ------
    TypeError
        if variables 'id_1' or 'id_2' are not strings.

    Returns
    ------
    df: dataframe
        dataframe with cluster number    

    Example
    -------
    >df.show()

    +---+---+
    |id1|id2|
    +---+---+
    | 1a| 2b|
    | 3a| 3b|
    | 2a| 1b|
    | 3a| 7b|
    | 1a| 8b|
    | 2a| 9b|
    +---+---+

    >linkage.cluster_number(df = df, id_1 = 'id1', id_2 = 'id2').show()

    +---+---+--------------+
    |id1|id2|Cluster_Number|
    +---+---+--------------+
    | 2a| 1b|             1|
    | 2a| 9b|             1|
    | 1a| 2b|             2|
    | 1a| 8b|             2|
    | 3a| 7b|             3|
    | 3a| 3b|             3|
    +---+---+--------------+     
    """
    # Check variable types
    if not ((isinstance(df.schema[id_1].dataType, StringType)) and (isinstance(df.schema[id_2].dataType, StringType))):
        raise TypeError('ID variables must be strings')

    # Set up spark checkpoint settings
    spark = SparkSession.builder.getOrCreate()
    username = os.getenv("HADOOP_USER_NAME")
    checkpoint_path = f"/user/{username}/checkpoints"
    spark.sparkContext.setCheckpointDir(checkpoint_path)

    # Stack all unique IDs datasets into one column called 'id'
    ids = df.select(id_1).union(df.select(id_2))
    ids = ids.select(id_1).distinct().withColumnRenamed(id_1, 'id')

    # Rename matched data columns ready for clustering
    matches = df.select(id_1, id_2).withColumnRenamed(
        id_1, 'src').withColumnRenamed(id_2, 'dst')

    # Create graph & get connected components / clusters
    try:
      graph = GraphFrame(ids, matches)
      
      cluster = graph.connectedComponents()

      # Update cluster numbers to be consecutive (1,2,3,4,... instead of 1,2,3,1000,1001...)
      lookup = cluster.select('component').dropDuplicates(['component']).withColumn(
          'Cluster_Number', F.rank().over(Window.orderBy("component"))).sort('component')
      cluster = cluster.join(lookup, on='component',
                             how='left').withColumnRenamed('id', id_1)

      # Join new cluster number onto matched pairs
      df = df.join(cluster, on=id_1, how='left').sort(
          'Cluster_Number').drop('component')

      return df
  
    except py4j.protocol.Py4JJavaError:
      print("""WARNING: A graphframes wrapper package installation has not been found! If you have not already done so,
      you will need to submit graphframes' JAR file dependency to your spark context. This can be found here:
      \nhttps://repos.spark-packages.org/graphframes/graphframes/0.6.0-spark2.3-s_2.11/graphframes-0.6.0-spark2.3-s_2.11.jar
      \nOnce downloaded, this can be submitted to your spark context via: spark.conf.set('spark.jars', path_to_jar_file)
      """)
###############################################################################


def extract_mk_variables(df, mk):
    '''
    Extracts variables from matchkey join condition

    For example, would return ['first_name','last_name',date_of_birth'] for a 
    matchkey using these components. Used in mk_drop(na) to exclude instances 
    of null in any matchkey component columns.

    Parameters
    ---------- 
    df : dataframe
      Dataframe the to which matchkeys will be applied.
    mk : list
      The join conditions as specified in matchkey

    Returns
    -------
    list
      list of components included in matchkey

    Raises
    -------
      None at present.  
    '''

    mk_variables = re.split("[^a-zA-Z0-9_]", str(mk))
    mk_variables = [x for x in mk_variables if x in df.columns]
    mk_variables = list(set(mk_variables))

    return mk_variables

###############################################################################


def mk_dropna(df, mk):
    """
    Drops null values in variables included in matchkeys join conditions

    Improves efficiency of join by excluding records containing nulls in matchkey 
    component columns as these records would not match. Also avoids skew resulting
    from nulls. Used in order_matchkeys() and matchkey_join()

    Parameters
    ---------- 
    df : dataframe
      Dataframe the to which matchkeys will be applied.
    mk : list
      A list of join conditions (as specified in a matchkey(s)).

    Returns
    -------
    dataframe
      Dataframe with null values dropped from matchkey component variables.

    Raises
    -------
      None at present.

    See Also
    --------
    extract_mk_variables()
    """
    variables = extract_mk_variables(df, mk)

    df = df.dropna(subset=variables)

    return df

###############################################################################


def order_matchkeys(df_l, df_r, mks, chunk=10):
    '''
    Orders matchkey components based on the number of matches made by each matchkey
    in ascending order

    Parameters
    ---------- 
    df : dataframe
      Dataframe the to which matchkeys will be applied.
    mk : list
      A list of join conditions (as specified in a matchkey(s)).

    Returns
    -------
    dataframe
      Dataframe with null values dropped from matchkey component variables.

    Raises
    -------
      None at present.

    '''
    mk_order = pd.DataFrame({
        'mks': mks,
        'supplied_order': [mk_n for mk_n, mk
                           in enumerate(mks)]
    })

    mks = chunk_list(mks, chunk)

    mk_counts = pd.DataFrame(columns=[
        'supplied_order',
        'count'
    ])

    chunk_n = 0-chunk

    for mk_chunk in mks:

        chunk_n += chunk

        df = da.union_all(*[
            (mk_dropna(df_l, mk).join(mk_dropna(df_r, mk),
                                      on=mk,
                                      how='inner')
             .withColumn('supplied_order', F.lit(mk_n+chunk_n))
             .select('supplied_order')
             )
            for mk_n, mk in enumerate(mk_chunk)
        ])

        df = (df
              .groupBy('supplied_order')
              .count()
              .toPandas()
              )

        mk_counts = (mk_counts
                     .append(df)
                     .reset_index(drop=True))

    mk_order = (mk_order
                .merge(mk_counts, on='supplied_order')
                .sort_values('count')
                )

    mk_order = list(mk_order['mks'])

    return mk_order

###############################################################################


def matchkey_join(df_l, df_r, id_l, id_r, mk, mk_n=0):
    """
    Joins dataframes on matchkey retaining only 1:1 matches

    Joins left and right dataframes on specified matchkey. Retains only instances
    of 1:1 matches between left and right identifiers (i.e. where matches are 
    unique and there is only one match candidate for each left and right identifier).
    Adds 'matchkey' column to record matchkey number.

    Parameters
    ---------- 
    df_l : dataframe
      left dataframe to be joined.
    df_r : dataframe
      right dataframe to be joined.
    id_l : string
      variable name of column containing left unique identifier
    id_r : string
      variable name of column containing right unique identifier
    mk : list
      matchkey join conditions
    mk_n : int
      matchkey number (order of application)

    Returns
    -------
    dataframe
      dataframe of unique 1:1 joins on left and right dataframe. Retaining
      only left and right identifiers and matchkey number.

    Raises
    -------
      None at present. 

    See Also
    --------
    mk_dropna()
    """
    #variables_l = extract_mk_variables(df_l,mk)
    #variables_r = extract_mk_variables(df_r,mk)

    #df_l = df_l.dropna(subset=variables_l)
    #df_r = df_r.dropna(subset=variables_r)

    df_l = mk_dropna(df_l, mk)
    df_r = mk_dropna(df_r, mk)

    df = (df_l
          .join(df_r, mk, 'inner')
          .select(id_l, id_r)
          .dropDuplicates()
          .withColumn('matchkey', F.lit(mk_n))
          )

    df = da.filter_window(df, id_r, id_l, 'count', 1)
    df = da.filter_window(df, id_l, id_r, 'count', 1)

    return df

###############################################################################


def chunk_list(l, n):
    return [l[i * n:(i + 1) * n]
            for i in range((len(l) + n - 1) // n)]

###############################################################################


def matchkey_dataframe(mks):
    """
    Creates dataframe of matchkeys and descriptions 

    Takes a list of matchkeys. Assigns numbers to matchkeys based on order in list 
    provided. Adds description of each matchkey from string manipulation of join
    condition.

    Parameters
    ---------- 
    mks : list
      list of matchkeys

    Returns
    -------
    dataframe
      Dataframe of matchkeys and descriptions.

    Raises
    -------
      None at present.  
    """
    spark = SparkSession.builder.getOrCreate()

    mk_df = (spark.createDataFrame(
        pd.DataFrame(
            {
                'matchkey': [x for x, y in enumerate(mks)],
                'description': [str(x) for x in mks],
            }
        )[['matchkey', 'description']]
    ).withColumn('description',
                 F.regexp_replace(
                     F.col('description'),
                     "(?:Column[<]b['])|(?:['][>][,] \
                 Column[<]b['])|(?:['][>])| ",
                     "")
                 ))

    return mk_df


###############################################################################


def assert_unique_matches(linked_ids, *identifier_col):
    """
    Asserts that all linkage results are unique (i.e. that there is 1:1 relationship 
    between matched records).

    Note: This will return an AssertError if linkage results are not unique.

    Parameters
    ---------- 
    linked_ids : dataframe
      linked dataframe that includes unique identifier columns
    identifier_col: string or multiple strings
      column name(s) of unique identifiers in linked data

    """

    for column in identifier_col:

        assert (linked_ids
                .groupBy(column)
                .count()
                .select(F.max(F.col('count')))
                .collect()[0][0]
                ) == 1

###############################################################################


def assert_unique(df, column):

    if type(column) != list:
        column = [column]

    assert df.count() == df.dropDuplicates(subset=column).count()

###############################################################################


def matchkey_counts(linked_df):
    """
    Counts number of links made on each matchkey

    Returns dataframe of matchkey number and the number of matches made on
    each matchkey.

    Parameters
    ---------- 
    linked_df : dataframe
      dataframe returned by deterministic_linkage(). This will include variables:
      left identifier; right identifier and matchkey number

    Returns
    -------
    dataframe
      Dataframe of counts of matches achieved by matchkey number.

    Raises
    -------
      None at present.    
    """

    return (linked_df
            .groupBy('matchkey')
            .count()
            .sort('count', ascending=False)
            )

###############################################################################


def clerical_sample(linked_ids, mk_df, df_l, df_r,
                    id_l, id_r, suffix_l='_l', suffix_r='_r',
                    n_ids=100):
    """
    Suffixes left and right dataframes with specified suffix. Joins raw data
    to linked identifier output of deterministic linkage. Returns a number of
    examples for each matchkey as specified.

    Parameters
    ---------- 
    linked_ids : dataframe
      dataframe returned by deterministic_linkage(). This will include variables:
      left identifier; right identifier and matchkey number.
    mk_df : dataframe
      dataframe returned by matchkey_dataframe(). This will include matchkey
      number and description
    df_l : dataframe
      left dataframe to be joined.
    df_r : dataframe
      right dataframe to be joined.
    id_l : string
      variable name of column containing left unique identifier
    id_r : string
      variable name of column containing right unique identifier
    suffix_l : string
      suffix to be applied to left dataframe
    suffix_r : string
      suffix to be applied to right dataframe
    n_ids : int, default = 100
      The number of identifier pairs sampled for each matchkey

    Returns
    -------
    dataframe
      Dataframe of deterministic linkage samples by matchkey.

    Raises
    -------
      None at present.    

    See Also
    --------
    dataframes.union_all()
    """

    mks = sorted([x[0] for x in
                  linked_ids.select('matchkey').dropDuplicates().collect()])

    linked_ids = (linked_ids
                  .withColumn('random', F.rand())
                  .sort('random')
                  .drop('random'))

    linked_ids = [(linked_ids
                   .where(F.col('matchkey') == mk)
                   .select('matchkey', id_l, id_r)
                   .dropDuplicates()
                   .limit(n_ids))
                  for mk in mks]

    linked_ids = da.union_all(*linked_ids)

    review_df = (linked_ids
                 .join(da.suffix_columns(df_l, suffix_l, exclude=id_l), id_l, 'inner')
                 .join(da.suffix_columns(df_r, suffix_r, exclude=id_r),
                       id_r, 'inner')
                 .join(mk_df, on='matchkey')
                 .sort('matchkey', id_l)
                 )

    lead_columns = ['matchkey', id_l, id_r]
    end_columns = ['description']

    review_df = (review_df
                 .select(lead_columns +
                         sorted([x for x in review_df.columns
                                 if x not in
                                 lead_columns+end_columns])
                         + end_columns)
                 )

    return review_df

###################################################################


def demographics(*args, df, identifier,):
    """
    Produces demographics count of dataframe for groups specified.

    Used to get counts of, for example number of persons in age 
    groups, to allow comparison of counts in raw data to counts in
    linked data. Produces output used in demographics_compare()

    Parameters
    ---------- 
    *args : str or list of str
      Variable column titles to be included in group counts
    df : dataframe
      the dataframe for which demographics are calculated
      number and description
    identifier : string
      Reference to column containing unique identifier

    Returns
    -------
    dataframe
      Dataframe of demograhic metrics.

    Raises
    -------
      None at present.

    See Also
    --------
    dataframes.union_all()
    """

    total_count = (df
                   .select(identifier)
                   .dropDuplicates()
                   ).count()

    df = da.union_all(*[
        (df
         .select(identifier, col)
         .dropDuplicates()
         .groupBy(col)
         .agg(F.count(F.col(identifier)).alias('count'))
         .fillna('not_provided')
         .withColumn('variable', F.lit(col))
         .withColumnRenamed(col, 'value')
         )
        for col in args
    ])

    df = (df
          .withColumn('total_count', F.lit(total_count))
          .select('variable', 'value', 'count', 'total_count')
          .sort(['variable', 'value'])
          ).coalesce(1)

    return df

####################################################################


def demographics_compare(df_raw, df_linked):
    """
    Compares raw and linked outputs produced by demographics() to
    highlight bias in linked data.

    Determines the expected counts in linked data per demographic
    group, from match rates and counts in raw data. Produces positive
    or negative discrepency metric, highlighting under or
    over-representation of each group in the linked data, versus the
    raw data.

    Parameters
    ---------- 
    df_raw : dataframe
      output of demographics() for raw data
    df_linked : dataframe
      output of demographics() for linked data

    Returns
    -------
    dataframe
      Dataframe of demographic comparison metrics.

    Raises
    -------
      None at present.
    """

    df_raw = (da.suffix_columns(df_raw,
                                '_raw',
                                exclude=['variable',
                                         'value']))

    df_linked = (da.suffix_columns(df_linked,
                                   '_linked',
                                   exclude=['variable',
                                            'value']))

    df = (df_raw
          .join(df_linked,
                on=['variable', 'value'],
                how='full')
          )

    df = df.fillna(0, subset=['count_linked', 'count_raw'])

    total_count_linked = int((df
                             .select('total_count_linked')
                             .dropna()
                             .dropDuplicates()
                             .toPandas()
                              )['total_count_linked'][0])
    df = df.withColumn('total_count_linked',
                       F.lit(total_count_linked))

    total_count_raw = int((df
                           .select('total_count_raw')
                           .dropna()
                           .dropDuplicates()
                           .toPandas()
                           )['total_count_raw'][0])
    df = df.withColumn('total_count_raw',
                       F.lit(total_count_raw))

    df = (df
          .withColumn('match_rate',
                      F.col('total_count_linked')
                      / F.col('total_count_raw')
                      )
          )

    df = (df
          .withColumn('proportional_count',
                      (F.col('match_rate')*F.col('count_raw'))
                      .cast('int')
                      )
          )

    df = (df
          .withColumn('proportional_discrepency',
                      (F.col('count_linked')-F.col('proportional_count'))
                      / F.col('proportional_count'))
          )

    df = (df
          .sort(['variable', 'value'])
          ).coalesce(1)

    return df

###############################################################################


def matchkeys_drop_duplicates(mks):
    ''' 
    Removes duplicates from generated matchkeys
    '''
    out = pd.DataFrame({'mks': mks})
    out['mks_str'] = [str(x) for x in out['mks']]
    out = out.drop_duplicates(subset=['mks_str'])
    out = list(out['mks'])

    return out

############################################################################

def deduplicate(df, record_id, mks):
    """
    Filters out duplicate records from a supplied dataframe.

    Parameters
    ---------- 
    df : dataframe
    record_id : string
      name of unique identifier column in data
    mks : list
      list of matchkeys

    Returns
    -------
    unique
      Dataframe of unique record ID pairs
    duplicates
      Dataframe of identified duplicate records

    Raises
    -------
      None at present.

    Example
    -------
    > CCS.count()
    10550

    > deduplicate_keys = [['First_Name','Last_Name', 'Resident_Age', 'Postcode'],
                         ['First_Name','Last_Name', 'Resident_Age','Postcode', 'Address']]

    > CCS = linkage.deduplicate(df = CCS, record_id = 'Resident_ID', mks = deduplicate_keys)[0]

    > CCS.count()
    10487
    """

    # get active spark session
    spark = SparkSession.builder.getOrCreate()

    # check to see if matchkeys are passed as a list of lists
    if any(isinstance(MK, list) for MK in mks) == False:
        mks = [mks]

    for count, MK in enumerate(mks, 1):

        if count == 1:

            unique = df.dropDuplicates(MK)

        else:

            unique = unique.dropDuplicates(MK)
    
    duplicates = df.join(unique, on = record_id, how = 'left_anti').dropDuplicates([record_id])


    return unique, duplicates

############################################################################

def deterministic_linkage(df_l, df_r, id_l, id_r, matchkeys, out_dir):
    '''
    Performs determistic linkage of two dataframes given a list of matchkeys /
    join conditions. Returns a dataframe of the linked identifers of left and 
    right dataframes,together with the numeric identifier of the matchkey / join 
    condition on which the link was achieved. Also saves a parquet of linked 
    identifiers in specified directory.

    Parameters
    ---------- 
    df_l : dataframe
      Left dtaframe to be linked
    df_r : dataframe
      Right dtaframe to be linked
    id_l : string
      Unique identifier in left dataframe
    id_r : string
      Unique identifier in right dataframe
    matchkeys : list
      A list of join conditions to be sequentially applied in linkage
    out_dir : string
      Specified file path for the directory in which parquet of linked identifiers
      will be saved and which will be used in processing the linkage. 

    Returns
    -------
    list
      a dataframe of the linked identifers of left and right dataframes,together with 
      the numeric identifier of the matchkey / join condition on which the link was 
      achieved. Also saves a parquet of linked identifiers in out_dir.

    Raises
    -------
      None at present.

    Example
    -------

    > MK1 = [CEN.Full_Name_cen == CCS.Full_Name_ccs,
            CEN.Sex_cen == CCS.Sex_ccs,
            CEN.Postcode_cen == CCS.Postcode_ccs]

    > MK2 = [CEN.First_Name_cen == CCS.First_Name_ccs,
             CEN.Last_Name_cen == CCS.Last_Name_ccs,
             CEN.Sex_cen == CCS.Sex_ccs,
             CEN.Postcode_cen == CCS.Postcode_ccs]

    > matchkeys = [MK1, MK2, MK3, MK4, MK5]

    > links = linkage.deterministic_linkage(df_l = CEN, df_r = CCS, 
                                            id_l = 'Resident_ID_cen', id_r = 'Resident_ID_ccs',
                                            matchkeys = matchkeys, out_dir = '/path_to_file')

    MATCHKEY 1
    matches on matchkey:  2815
    total matches:  2815
    left residual:  997186
    right residual:  7663

    MATCHKEY 2
    matches on matchkey:  384
    total matches:  3199
    left residual:  996802
    right residual:  7279

    > links.show()
    +--------------------+--------------------+--------+
    |     Resident_ID_cen|     Resident_ID_ccs|matchkey|
    +--------------------+--------------------+--------+
    |C1075168650487354680|C5417150230708747120|       1|
    |C1111234343783150025|C6146302309226089123|       1|
    |C1338540771296365051|C8521797154702755129|       1|
    |C1604818046072784138|C3975523078369491788|       1|
    +--------------------+--------------------+--------+
    '''

    # control for file path format
    if out_dir[-1] == "/":
        out_dir = out_dir[:-1]

    # count of unique ids in left df
    df_l_count = (df_l
                  .select(id_l)
                  .drop_duplicates()
                  ).count()
    # count of unique ids in right df
    df_r_count = (df_r
                  .select(id_r)
                  .drop_duplicates()
                  ).count()
    # initial count of matches
    count = 0

    for index, matchkey in enumerate(matchkeys, 1):

        if index == 1:
            # writes first matchkey to parquet
            ut.write_format(matchkey_join(
                df_l, df_r, id_l, id_r, matchkey, index),
                'parquet',
                f"{out_dir}/linked_identifiers",
                mode='overwrite')

        else:
            # reads previous matches
            # used in left anti join to ignore matched records
            matches = ut.read_format('parquet',
                                    f"{out_dir}/linked_identifiers")

            last_count = count
            count = matches.count()

            print("\nMATCHKEY", index-1)
            print("matches on matchkey: ", count-last_count)
            print("total matches: ", count)
            print("left residual: ", df_l_count-count)
            print("right residual: ", df_r_count-count)

            # appends subsequent matches to initial parquet
            ut.write_format(matchkey_join(
                df_l.join(matches, id_l, 'left_anti'),
                df_r.join(matches, id_r, 'left_anti'),
                id_l, id_r, matchkey, index),
                'parquet',
                f"{out_dir}/linked_identifiers",
                mode='append')

    # reads and returns final matches
    matches = ut.read_format('parquet',
                            f"{out_dir}/linked_identifiers")

    last_count = count
    count = matches.count()

    print("\nMATCHKEY", index)
    print("matches on matchkey: ", count-last_count)
    print("total matches: ", count)
    print("left residual: ", df_l_count-count)
    print("right residual: ", df_r_count-count)

    return matches
