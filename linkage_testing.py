import pyspark
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import pandas as pd
from dlh_utils.standardisation import *
from dlh_utils.dataframes import *
from dlh_utils.linkage import *


def test_order_matchkeys():
    spark = SparkSession.builder.getOrCreate()
    dfo = spark.createDataFrame(
        (pd.DataFrame({
            "uprn": ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20'],

            "first_name": ['aa', 'ba', 'ab', 'bb', 'aa', 'ax', 'cr', 'cd', 'dc', 'dx',
                           'ag', 'rd', 'rf', 'rg', 'rr', 'dar', 'dav', 'dam', 'dax', 'dev'],

            "last_name": ['fr', 'gr', 'fa', 'ga', 'gx', 'mx', 'ra', 'ga', 'fg', 'gx', 'mr',
                          'pr', 'ar', 'to', 'lm', 'pr', 'pf', 'se', 'xr', 'xf']
        })))

    dffn = spark.createDataFrame(
        (pd.DataFrame({
            "uprn": ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20'],

            "first_name": ['ax', 'bx', 'ad', 'bd', 'ar', 'ax', 'cr', 'cd', 'dc', 'dx',
                           'ag', 'rd', 'rf', 'rg', 'rr', 'dar', 'dav', 'dam', 'dax', 'dev'],

            "last_name": ['fr', 'gr', 'fa', 'ga', 'gx', 'mx', 'ra', 'ga', 'fg', 'gx', 'mr',
                          'pr', 'ar', 'to', 'lm', 'pr', 'pf', 'se', 'xr', 'xf']
        })))
    mks = [
        [F.substring(dfo['first_name'], 1, 1) == F.substring(dffn['first_name'], 1, 1),
         F.substring(dfo['last_name'], 1, 1) == F.substring(dffn['last_name'], 1, 1)],

        [F.substring(dfo['first_name'], 1, 1) == F.substring(dffn['first_name'], 1, 1),
            dfo['last_name'] == dffn['last_name']],
        [dfo['first_name'] == dffn['first_name'],
            dfo['last_name'] == dffn['last_name']]
    ]

    testDf = pd.DataFrame({
        'mks': mks,
        'count': [(dfo.join(dffn, on=mk, how='inner')).count()
                  for mk in mks]
    })
    testDf = testDf.sort_values('count')
    testDf['mks'] = [str(x) for x in testDf['mks']]

    test_mks = list(testDf['mks'])

    [str(x) for x in order_matchkeys(dfo, dffn, mks)]

    assert ([str(x) for x in order_matchkeys(dfo, dffn, mks)]
            == test_mks)

    ###########################################################################


def test_matchkey_join():
    spark = SparkSession.builder.getOrCreate()
    df_l = spark.createDataFrame(
        (pd.DataFrame({
            "l_id": ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20'],

            "first_name": ['aa', 'ba', 'ab', 'bb', 'aa', 'ax', 'cr', 'cd', 'dc', 'dx',
                           'ag', 'rd', 'rf', 'rg', 'rr', 'dar', 'dav', 'dam', 'dax', 'dev'],

            "last_name": ['fr', 'gr', 'fa', 'ga', 'gx', 'mx', 'ra', 'ga', 'fg', 'gx', 'mr',
                          'pr', 'ar', 'to', 'lm', 'pr', 'pf', 'se', 'xr', 'xf']
        })))

    df_r = spark.createDataFrame(
        (pd.DataFrame({
            "r_id": ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20'],

            "first_name": ['ax', 'bx', 'ad', 'bd', 'ar', 'ax', 'cr', 'cd', 'dc', 'dx',
                           'ag', 'rd', 'rf', 'rg', 'rr', 'dar', 'dav', 'dam', 'dax', 'dev'],

            "last_name": ['fr', 'gr', 'fa', 'ga', 'gx', 'mx', 'ra', 'ga', 'fg', 'gx', 'mr',
                          'pr', 'ar', 'to', 'lm', 'pr', 'pf', 'se', 'xr', 'xf']
        })))

    mks = [
        [df_l['first_name'] == df_r['first_name'],
         df_l['last_name'] == df_r['last_name']],

        [F.substring(df_l['first_name'], 1, 1) == F.substring(df_r['first_name'], 1, 1),
            df_l['last_name'] == df_r['last_name']],

        [F.substring(df_l['first_name'], 1, 1) == F.substring(df_r['first_name'], 1, 1),
            F.substring(df_l['last_name'], 1, 1) == F.substring(df_r['last_name'], 1, 1)]
    ]

    assert (matchkey_join(df_l, df_r, 'l_id', 'r_id', mks[2], 1).count() == 10)
    assert (matchkey_join(df_l, df_r, 'l_id', 'r_id', mks[1], 1).count() == 20)
    assert (matchkey_join(df_l, df_r, 'l_id', 'r_id', mks[0], 1).count() == 15)

####################################################################


def test_extract_mk_variables():
    spark = SparkSession.builder.getOrCreate()
    df_l = spark.createDataFrame(
        (pd.DataFrame({
            "l_id": ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20'],

            "first_name": ['aa', 'ba', 'ab', 'bb', 'aa', 'ax', 'cr', 'cd', 'dc', 'dx',
                           'ag', 'rd', 'rf', 'rg', 'rr', 'dar', 'dav', 'dam', 'dax', 'dev'],

            "last_name": ['fr', 'gr', 'fa', 'ga', 'gx', 'mx', 'ra', 'ga', 'fg', 'gx', 'mr',
                          'pr', 'ar', 'to', 'lm', 'pr', 'pf', 'se', 'xr', 'xf']
        })))

    mks = [
        [df_l['first_name'] == df_r['first_name'],
         df_l['last_name'] == df_r['last_name']],

        [F.substring(df_l['first_name'], 1, 1) == F.substring(df_r['first_name'], 1, 1),
            df_l['last_name'] == df_r['last_name']],

        [F.substring(df_l['first_name'], 1, 1) == F.substring(df_r['first_name'], 1, 1),
            F.substring(df_l['last_name'], 1, 1) == F.substring(df_r['last_name'], 1, 1)]
    ]
    assert (extract_mk_variables(df_l, mks)[0] == 'first_name')
    assert (extract_mk_variables(df_l, mks)[1] == 'last_name')

###########################################################################


def test_deterministic_linkage():
    spark = SparkSession.builder.getOrCreate()
    df_l = spark.createDataFrame(
        (pd.DataFrame({
            "l_id": ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20'],

            "first_name": ['aa', 'ba', 'ab', 'bb', 'aa', 'ax', 'cr', 'cd', 'dc', 'dx',
                           'ag', 'rd', 'rf', 'rg', 'rr', 'dar', 'dav', 'dam', 'dax', 'dev'],

            "last_name": ['fr', 'gr', 'fa', 'ga', 'gx', 'mx', 'ra', 'ga', 'fg', 'gx', 'mr',
                          'pr', 'ar', 'to', 'lm', 'pr', 'pf', 'se', 'xr', 'xf']
        })))

    df_r = spark.createDataFrame(
        (pd.DataFrame({
            "r_id": ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20'],

            "first_name": ['ax', 'bx', 'ad', 'bd', 'ar', 'ax', 'cr', 'cd', 'dc', 'dx',
                           'ag', 'rd', 'rf', 'rg', 'rr', 'dar', 'dav', 'dam', 'dax', 'dev'],

            "last_name": ['fr', 'gr', 'fa', 'ga', 'gx', 'mx', 'ra', 'ga', 'fg', 'gx', 'mr',
                          'pr', 'ar', 'to', 'lm', 'pr', 'pf', 'se', 'xr', 'xf']
        })))

    mks = [
        [df_l['first_name'] == df_r['first_name'],
         df_l['last_name'] == df_r['last_name']],

        [F.substring(df_l['first_name'], 1, 1) == F.substring(df_r['first_name'], 1, 1),
            df_l['last_name'] == df_r['last_name']],

        [F.substring(df_l['first_name'], 1, 1) == F.substring(df_r['first_name'], 1, 1),
            F.substring(df_l['last_name'], 1, 1) == F.substring(df_r['last_name'], 1, 1)]
    ]

    assert ((deterministic_linkage(df_l, df_r, 'l_id', 'r_id', mks)
             .filter(F.col('l_id') <= 5))
            .where(F.col('matchkey') == 1)
            .count() == 5)

    assert ((deterministic_linkage(df_l, df_r, 'l_id', 'r_id', mks)
             .filter(F.col('l_id') > 5))
            .where(F.col('matchkey') == 0)
            .count() == 15)

##################################################################


def test_demographics():

    spark = SparkSession.builder.getOrCreate()

    df_raw = spark.createDataFrame(
        (pd.DataFrame({
            "id": [x for x in range(40)],
            "sex": (['M']*20)+(['F']*20),
            "age_group": (['10-20']*10)+(['20-30']*10)+(['30-40']*10)+(['50-60']*10),
        })))

    df_linked = spark.createDataFrame(
        (pd.DataFrame({
            "id": [x for x in range(20)],
            "sex": (['M']*10)+(['F']*10),
            "age_group": (['10-20']*5)+(['20-30']*5)+(['30-40']*5)+(['50-60']*5),
        })))

    dem_raw = demographics(*['sex', 'age_group'],
                           df=df_raw, identifier='id').toPandas() ==\
        spark.createDataFrame(
        [('age_group', '10-20', 10, 40),
         ('age_group', '20-30', 10, 40),
            ('age_group', '30-40', 10, 40),
            ('age_group', '50-60', 10, 40),
            ('sex', 'F', 20, 40),
            ('sex', 'M', 20, 40)],
        ['variable', 'value', 'count', 'total_count']
    ).toPandas()

    assert [dem_raw[dem_raw[variable] != True].shape[0]
            for variable in list(dem_raw)] == [0, 0, 0, 0]

    dem_linked = demographics(*['sex', 'age_group'],
                              df=df_linked, identifier='id').toPandas() ==\
        spark.createDataFrame(
        [('age_group', '10-20', 5, 20),
         ('age_group', '20-30', 5, 20),
            ('age_group', '30-40', 5, 20),
            ('age_group', '50-60', 5, 20),
            ('sex', 'F', 10, 20),
            ('sex', 'M', 10, 20)],
        ['variable', 'value', 'count', 'total_count']
    ).toPandas()

    assert [dem_linked[dem_linked[variable] != True].shape[0]
            for variable in list(dem_linked)] == [0, 0, 0, 0]

##################################################################


def test_demographics_compare():

    spark = SparkSession.builder.getOrCreate()

    df_raw = spark.createDataFrame(
        (pd.DataFrame({
            "id": [x for x in range(40)],
            "sex": (['M']*20)+(['F']*20),
            "age_group": (['10-20']*10)+(['20-30']*10)+(['30-40']*10)+(['50-60']*10),
        })))

    df_linked = spark.createDataFrame(
        (pd.DataFrame({
            "id": [x for x in range(20)],
            "sex": (['M']*10)+(['F']*10),
            "age_group": (['10-20']*5)+(['20-30']*5)+(['30-40']*5)+(['50-60']*5),
        })))

    dem_raw = demographics(*['sex', 'age_group'],
                           df=df_raw, identifier='id')

    dem_linked = demographics(*['sex', 'age_group'],
                              df=df_linked, identifier='id')

    assert (demographics_compare(dem_raw, dem_linked)
            .where(F.col('proportional_discrepency') != 0)
            ).count() == 0

    df_raw = spark.createDataFrame(
        (pd.DataFrame({
            "id": [x for x in range(40)],
            "sex": (['M']*20)+(['F']*20),
            "age_group": (['10-20']*10)+(['20-30']*10)+(['30-40']*10)+(['50-60']*10),
        })))

    df_linked = spark.createDataFrame(
        (pd.DataFrame({
            "id": [x for x in range(20)],
            "sex": (['M']*15)+(['F']*5),
            "age_group": (['10-20']*5)+(['20-30']*5)+(['30-40']*5)+(['50-60']*5),
        })))

    dem_raw = demographics(*['sex', 'age_group'],
                           df=df_raw, identifier='id')

    dem_linked = demographics(*['sex', 'age_group'],
                              df=df_linked, identifier='id')

    assert (demographics_compare(dem_raw, dem_linked)
            .where(F.col('proportional_discrepency') != 0)
            ).count() == 2

    assert sorted([x[0] for x in
                   (demographics_compare(dem_raw, dem_linked)
                    .where(F.col('variable') == 'sex')
                    .select('proportional_discrepency')
                    ).collect()]) == [-0.5, 0.5]

##############################################################


def test_assert_unique_matches():

    spark = SparkSession.builder.getOrCreate()

    df = spark.createDataFrame(
        (pd.DataFrame({
            "id_l": ['1', '2', '3', '4', '5'],
            "id_r": ['a', 'b', 'c', 'd', 'e'],
        })))

    x = 0
    try:
        assert_unique_matches(df, 'id_l', 'id_r')
    except:
        x = 1
    assert x == 0

    df = spark.createDataFrame(
        (pd.DataFrame({
            "id_l": ['1', '1', '3', '4', '5'],
            "id_r": ['a', 'b', 'c', 'd', 'd'],
        })))

    x = 0
    try:
        assert_unique_matches(df, 'id_l', 'id_r')
    except:
        x = 1
    assert x == 1


##############################################################

def test_matchkey_counts():

    spark = SparkSession.builder.getOrCreate()

    df = spark.createDataFrame(
        (pd.DataFrame({
            "matchkey": ['1', '1', '3', '4', '4'],
            "id_r": ['a', 'b', 'c', 'd', 'e'],
        })))

    df = (matchkey_counts(df)
          .toPandas()
          .sort_values('matchkey')
          )

    assert list(df['matchkey']) == ['1', '3', '4']
    assert list(df['count']) == [2, 1, 2]

##############################################################


def test_matchkey_dataframe():

    spark = SparkSession.builder.getOrCreate()

    df_l = spark.createDataFrame(
        (pd.DataFrame({
            "first_name": ['test']*10,
            "last_name": ['test']*10,
            "uprn": ['test']*10,
            "date_of_birth": ['test']*10,
        })))

    df_r = spark.createDataFrame(
        (pd.DataFrame({
            "first_name": ['test']*10,
            "last_name": ['test']*10,
            "uprn": ['test']*10,
            "date_of_birth": ['test']*10,
        })))

    mks = [
        [
            df_l['first_name'] == df_r['first_name'],
            df_l['last_name'] == df_r['last_name'],
            df_l['uprn'] == df_r['uprn'],
            df_l['date_of_birth'] == df_r['date_of_birth'],
        ],
        [
            F.substring(df_l['first_name'], 0, 2) == F.substring(
                df_r['first_name'], 0, 2),
            F.substring(df_l['last_name'], 0, 2) == F.substring(
                df_r['last_name'], 0, 2),
            df_l['uprn'] == df_r['uprn'],
            df_l['date_of_birth'] == df_r['date_of_birth'],
        ]
    ]

    assert list(matchkey_dataframe(mks).toPandas()['description']) ==\
        ['[(first_name=first_name),(last_name=last_name),(uprn=uprn),'
         + '(date_of_birth=date_of_birth)]',
         '[(substring(first_name,0,2)=substring(first_name,0,2)),'
         + '(substring(last_name,0,2)=substring(last_name,0,2)),(uprn=uprn),'
         + '(date_of_birth=date_of_birth)]']

    assert list(matchkey_dataframe(mks).toPandas()['matchkey']) ==\
        [0, 1]

    df_l = spark.createDataFrame(
        (pd.DataFrame({
            "id_l": [-1, -2, -3],
            "first_name": ['AMY', 'AMY', 'AMY'],
            "last_name": ['SMITH', 'SMITH', 'SMITH'],
            "date_of_birth": ['a', None, 'b'],
            "uprn": ['a', 'b', None],
            "sex": ['F', 'F', 'F']
        })))

    df_r = spark.createDataFrame(
        (pd.DataFrame({
            "id_r": [-1, -2, -3],
            "first_name": ['AMY', 'AMY', 'AMY'],
            "last_name": ['SMITH', 'SMITH', 'SMITH'],
            "date_of_birth": ['a', None, 'b'],
            "uprn": ['a', 'b', None],
            "sex": ['F', 'F', 'F']
        })))

    mks = [
        [
            df_l['first_name'] == df_r['first_name'],
            df_l['last_name'] == df_r['last_name'],
            df_l['sex'] == df_r['sex'],
            df_l['uprn'] == df_r['uprn'],
            df_l['date_of_birth'] == df_r['date_of_birth'],
        ],
        [
            df_l['first_name'] == df_r['first_name'],
            df_l['last_name'] == df_r['last_name'],
            df_l['sex'] == df_r['sex'],
            df_l['uprn'] == df_r['uprn'],
        ],
        [
            df_l['first_name'] == df_r['first_name'],
            df_l['last_name'] == df_r['last_name'],
            df_l['sex'] == df_r['sex'],
            df_l['date_of_birth'] == df_r['date_of_birth'],
        ],
    ]

    assert (li.deterministic_linkage(df_l, df_r, 'id_l', 'id_r', mks)
            .where(F.col('id_l') != F.col('id_r'))).count() == 0
