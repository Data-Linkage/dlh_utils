'''
Functions used to add visual formatting to Pandas dataframes and export them to Excel.
These wrap up some commonly-needed Pandas Styler and openpyxl boilerplate code.
'''
import os
import subprocess
import openpyxl
import pandas as pd

##################################################################################


def export_to_excel(
    dataframes,
    styles=None,
    columns=None,
    freeze_panes=None,
    local_path=None,
    hdfs_path=None
):
    """
      Creates an Excel workbook with one worksheet for each of the provided
      dataframes.

      Parameters
      ----------
      dataframes : dict, list or dataframe
        A dictionary whose keys are names for the datasheets and values are
        Pandas dataframes, or just a list of dataframes; in the latter case
        the function will name the sheets Sheet1, Sheet2 etc.
        If pyspark dataframes are provided they will be converted to Pandas.
        A single dataframe can also be passed for this argument.
      styles : dictionary, default=None
        A dictionary to pass to apply_excel_styles(). See the documentation
        for that function for more information.
      columns : list or dictionary of lists, default=None
        A dictionary whose keys are dataframe names and whose values are lists
        of columns. If a dataframe is named in this dictionary, only the listed
        columns will be written to Excel and their order will be as in the list
        provided. If a dataframe is not named in this dictionary, all of its
        columns will be included in their default order.
      freeze_panes : dictionary, default = None
        Dictionary mapping table names to tuples of the form (r, c) where r is
        the number of rows from the top to freeze and c the number of columns
        on the left. If a table's name is not present as a key, nothing will
        be frozen.
      local_path : string, default=None
        Full path (including filename) where the Excel workbook will be saved.
        If not specified, the workbook will not be saved to disk.
      hdfs_path : string, default=None
        Full HDFS path (including filename) where the Excel workbook will be saved.
        If you specify this, you must also provide a local_path.

      Returns
      -------
      An openpyxl.WorkBook object

      Example
      -------

      Write two dataframes to named sheets, selecting only two columns from people_df:

      > write_excel(
        {'People': people_df, 'Places': places_df},
        '/tmp/abc.xlsx',
        columns={'People': ['surname', firstname']}
        )

      This next example shows a single dataframe with complex styling:
        * bank_balance is given a background colour gradient
        * income_change is given red/green text colours for negative/positive
        * age is formatted in bold if the person is a child
      Note the use of partial() to set values for the parameters when we
      don't want to use their default values.

      See apply_styles and the style_* functions for more information on
      these functions.

      > from functools import partial
      > write_excel(
          {'People': people_df},
          '/tmp/abc.xlsx',
          styles={
            partial(
                style_colour_gradient,
                min=df["bank_balance"].min(),
                max=df["bank_balance"].max()
                ) : 'bank_balance',
            partial(
                style_on_cutoff,
                property='color'
                ):'income_change',
            partial(
              style_on_condition,
              condition=lambda x : x < 18
              ): "age"
          }
        )

    """

    if hdfs_path is not None and local_path is None:
        raise ValueError("Can't save to HDFS without also specifying a local path")
    if columns is None:
        columns = {}
    if isinstance(dataframes, list):
        dataframes = {"Sheet" + str(i): df for i, df in enumerate(dataframes)}
    elif isinstance(dataframes, pd.DataFrame):
        dataframes = {"Sheet1": dataframes}

    if isinstance(columns, list):
        if len(dataframes) > 1:
            raise ValueError(
                "Can't pass a list of columns to write_excel "
                "unless you only passed in a single dataframe. "
                "You can use a dictionary instead "
                "(see this function's docstring for an example)"
            )
        columns = {"Sheet1": columns}

    # Set up the workbook
    wb = openpyxl.Workbook()
    wb.save(local_path)
    for df_name in dataframes:
        wb.create_sheet(df_name)

    # Create a writer to export the dataframes
    writer = pd.ExcelWriter(
        local_path,
        mode="w",
        engine="openpyxl"
    )
    writer.book = wb
    writer.sheets = dict((ws.title, ws) for ws in wb.worksheets)

    # Export each dataframe to its own sheet
    for df_name in dataframes:
        if not isinstance(dataframes[df_name], pd.DataFrame):
            dataframes[df_name] = dataframes[df_name].toPandas()
        if df_name in columns:
            dataframes[df_name] = dataframes[df_name].loc[:,columns[df_name]]
        if styles is not None and df_name in styles:
            df_export = apply_styles(dataframes[df_name], styles[df_name])
        else:
            df_export = dataframes[df_name]
        fp = None
        if freeze_panes is not None and df_name in freeze_panes:
            fp = freeze_panes[df_name]
        df_export.to_excel(
            writer,
            sheet_name=df_name,
            index=False,
            freeze_panes=fp
        )

    # Remove the default, empty sheet if it wasn't used
    if "Sheet" not in dataframes:
        del wb["Sheet"]

    # Save to disk
    if local_path is not None:
        wb.save(local_path)
        if hdfs_path is not None:
            copy_local_file_to_hdfs(local_path, hdfs_path)

    return wb

#########################################################


def copy_local_file_to_hdfs(
    local_path,
    hdfs_path,
    local_filename=None,
    hdfs_filename=None
):
    """
    Copies a file created locally (i.e. in CDSW) to HDFS.

    Parameters
    ----------

    local_path: string
      Path to the local file to be copied
    hdfs_path: string
      Target path to copy to
    local_filename: string, default=None
      If not specified, the local_path is assumed to include the filename
    hdfs_filename: string, default=None
      If not specified, the hdfs_path is assumed to include the filename

    Returns
    -------
    None

    Example
    -------

    > copy_local_file_to_hdfs('/tmp/wb.xlsx', '/hdfs/folder/xlsx)
    """
    if local_filename is not None:
        local_path = os.path.join(local_path, local_filename)
    if hdfs_filename is not None:
        hdfs_path = os.path.join(hdfs_path, hdfs_filename)
    commands = ["hadoop", "fs", "-put", "-f", local_path, hdfs_path]
    process = subprocess.Popen(
        commands,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    stdout, stderr = process.communicate()

###############################################################################


def apply_styles(df, styles):
    """
      Applies a set of custom style functions to a dataframe and returns
      a pandas Styler object, which can be displayed by Jupyter and saved
      into Excel or HTML.

      Some suitable functions are included in this module with the prefix
      "style_" but you can also pass in a custom function.

      NOTE: This function returns a Styler, not a DataFrame.

      Parameters
      ----------
      df : Pandas dataframe
        The dataframe to be styled.
        If a pyspark dataframe is provided it will be converted to Pandas.
      styles : dict
        A dictionary whose keys are functions and whose values are lists
        of column names. Each function should take in a single value and
        return a valid CSS string. The value can be a single column name
        (as a string) or a list

      Returns
      -------
      A pandas Styler object.

      Example
      -------

      The DataFrame df has a column, "Number", that can be positive or negative.
      We apply the default style to a column of this type using a style_ function
      defined in this module:

      > apply_styles(
          df,
          {
            style_on_cutoff: "Number"
          }
        )

      We would like to highlight in bold when a value is NA in two columns,
      "Number" and "OtherNumber". Both style rules will be applied to the "Number"
      column but only the bold style to "OtherNumber". Again, we use a style_
      function defined in this module:
      > apply_styles(
          df,
          {
            style_on_cutoff: "Number",
            style_on_condition: ["Number", "OtherNumber"]
          }
        )

      The style_ functions have default behaviours we may want to customize. To
      do this, use the following pattern. The partial() function is defined in
      functools (part of python's standard library) and allows us to "freeze" some
      parameters of a function before it's evaluated:
      > from functools import partial
      > apply_styles(
          df,
          {
            partial(style_fill_pos_neg, property='color'): "Number"
          }
        )
    """

    if not isinstance(df, pd.DataFrame):
        df = df.toPandas()

    styles_by_column = {c:[] for c in df.columns}
    for s in styles:
        cols = styles[s]
        if not isinstance(cols, list):
            cols = [cols]
        for c in cols:
            styles_by_column[c].append(s)

    sdf = df.style
    for col, sty in styles_by_column.items():
        if len(sty) > 0:
            for f in sty:
                sdf = sdf.applymap(f, subset=col)
    return sdf

###############################################################################


def style_on_cutoff(
    value,
    cutoff=0,
    negative_style="red",
    positive_style="green",
    zero_style="white",
    error_style="black",
    property="background-color"
):
    """
      Returns a CSS string that sets the specified property to the appropriate style
      for the value passed in. The style is chosen based on whether the value is
      greater than, equal to or less than the cutoff.

      By default the cutoff is 0 and the function assigns background colours:
      green for positive, red for negative and white for exactly zero.

      You can also set a style for if the attempt to calculate a result led to an
      exception or if value < cutoff, value > cutoff and value == cutoff all
      evaluate to False.

      This function is intended to be used with apply_styles() (defined in this module).

      Parameters
      ----------

      value : numeric or other appropriate type
        A value of any type that can be compared with cutoff using "<" and ">".
      cutoff : any type comparable to value
        The passed-in value will be compared to cutoff to determine which style is returned.
      negative_style : string, default="red"
        The colour name, RGB code or other style value to be assigned when value < cutoff.
      positive_style : string, default="green"
        The colour name, RGB code or other style value to be assigned when value > cutoff.
      zero_style : string, default="white"
        The colour name, RGB code or other style value to be assigned when neither
        value < cutoff nor value > cutoff.
      error_style : string, default="black"
       The colour name, RGB code or other style value to be assigned when an error occurs.
       This can happen when the value is NaN or not of the expected type.
       If error_style=None, the exception will be re-raised instead. If in doubt, pass in
       None and make sure any errors that are raised are expected.
      property : string, default="background-color"
        The CSS property the colour will be applied to.

      Returns
      -------
      String
    """
    try:
        if value < cutoff:
            return property + " : " + negative_style + ";"
        elif value > cutoff:
            return property + " : " + positive_style + ";"
        elif value == cutoff:
            return property + " : " + zero_style + ";"
        else:
            raise ValueError(
                "Value " + str(value)
                + " was not less than, equal to or greater than cutoff "
                + str(cutoff)
            )
    except Exception as ex:
        if error_style is None:
            raise ex
        return property + " : " + error_style + ";"

###############################################################################


def style_on_condition(
    value,
    property="font-weight",
    true_style="bold",
    false_style="normal",
    error_style=None,
    condition=lambda x:x == 0
):
    """
      Returns a CSS string that sets the specified property to the appropriate style
      for the value passed in.

      This function is intended to be used with apply_styles() (defined in this module).

      Parameters
      ----------

      value : any appropriate type
        A value of a type that can be accepted by the condition function.
      property : string, default="font-weight"
        The CSS property the style will be applied to.
      true_style : string, default="bold"
        The style will be assigned when the condition evaluates true on the value.
      false_style : string, default="normal"
        The style will be assigned when the condition evaluates false on the value.
      error_style : string, default=None
        The style will be assigned if an error occurs in this function. If None,
        the error will be raised instead.
      condition : function, default=lambda x : x == 0
        A function that accepts value and returns a truthy value. This is used
        to determine whether the current value receives true_style or false_style.
        The default function applies true_style to values that exactly equal zero.

      Returns
      -------
      String
    """
    try:
        if condition(value):
            return property + " : " + true_style + ";"
        return property + " : " + false_style + ";"
    except Exception as ex:
        if error_style is None:
            raise ex
        return property + " : " + error_style + ";"

###############################################################################


def style_colour_gradient(
    value,
    min,
    max,
    property="background-color",
    min_colour="#FFFFFF",
    max_colour="#FF0000",
    error_colour="#000000"
):
    """
      Returns a CSS string that sets the specified colour property to a colour ranging
      between start_colour and end_colour depending on the value's position in the range
      between min and max.

      This function is intended to be used with apply_styles() (defined in this module).

      Parameters
      ----------

      value : any numeric type
        The value to be mapped to a colour.
      min : any numeric type
        The highest value that will receive the start_colour (any lower values
        also receive start_colour).
      max : any numeric type
        The lowest value that will receive the end_colour (any higher values
        also receive end_colour).
      property : string, default="background-colour"
        The CSS property the style will be applied to. Must be able to be set to
        a hexadecimal colour string.
      min_colour : string, default="#FFFFFF" (white)
        The colour at the minimum end of the gradient. Pass only a hexadecimal string,
        not a colour name.
      max_colour : string, default="#FF0000" (red)
        The colour at the maximum end of the gradient. Pass only a hexadecimal string,
        not a colour name.
      error_colour : string, default="#000000"
        The colour will be assigned if an error occurs in this function. If None,
        the error will be raised instead.

      Returns
      -------
      String
    """

    try:
        # Extract colour channels from parameters
        min_colour = min_colour.replace("#", "")
        max_colour = max_colour.replace("#", "")
        min_channels = [int(min_colour[i:i + 2], 16) for i in (0, 2, 4)]
        max_channels = [int(max_colour[i:i + 2], 16) for i in (0, 2, 4)]

        # Interpolate
        position = (value - min) / (max - min)
        interpolated_channels = [0, 0, 0]
        for c in range(3):
            if max_channels[c] > min_channels[c]:
                val = int(position * (max_channels[c] - min_channels[c]) + min_channels[c])
            else:
                val = int((1 - position) * (min_channels[c] - max_channels[c]) + max_channels[c])
            interpolated_channels[c] = ("0x%0*x" % (2, val))[2:].upper()

        # Return the result
        return property + " : #" + "".join(interpolated_channels) + ";"

    except Exception as ex:
        if error_colour is None:
            raise ex
        return property + " : #" + error_colour + ";"

###############################################################################


def style_map_values(
    value,
    mapping_dictionary,
    property="background-color",
    default_style=None,
    error_style=None
):
    """
      Returns a CSS string that sets the specified property to a value as specified by
      mapping_dictionary, which maps possible values being passed in to the style the
      property should be assigned to. If the value is not found in mapping_dictionary
      the default_value is used, if one is specified, or an error is raised. In the
      event of an error, error_value will be used if it is not None, otherwise the
      caller will receive the error.

      This function is intended to be used with apply_styles() (defined in this module).

      Parameters
      ----------

      value : any appropriate type
        A value of a type that can be accepted by the condition function.
      mapping_dictionary : dictionary
        Keys are possible values for the parameter "value"; these are mapped to styles.
      property : string, default="background-colour"
        The CSS property the style will be applied to. Must be able to be set to
        a hexadecimal colour string.
      default_style : string, default=None
        If not None, this will be used if the value passed in is not found in
        mapping_dictionary.
      error_style : string, default=None
        The style will be assigned if an error occurs in this function. If None,
        the error will be raised instead.

      Returns
      -------
      String
    """

    try:
        if value in mapping_dictionary:
            style = mapping_dictionary[value]
        elif default_style is not None:
            style = default_style
        else:
            str_val = str(value)
            raise ValueError(
                f"Value {str_val} not found in "
                "mapping_dictionary and no default_value "
                "was specified."
            )
        # Return the result
        return property + " : " + style + ";"

    except Exception as ex:
        if error_style is None:
            raise ex
        return property + " : " + error_style + ";"
