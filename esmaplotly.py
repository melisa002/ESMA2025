import plotly.express as px
import pandas as pd # for get_dict_colors

#from src.Automatic_report.pyAutoMonitoring import get_dict_colors

#----------------------------------------------------------------
# validate functions/error handling
# to raise messages for most common errors
#----------------------------------------------------------------

# is column in dataframe? if not, raise error
# valid for a list of columns
def validate_columns(df, columns):
    for column in columns:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in the DataFrame.")

# functions to check object types and error handling
def validate_data_frame(data_frame):
    if not isinstance(data_frame, pd.DataFrame):
        raise ValueError(f"Input data_frame must be a pandas DataFrame.")

def validate_string(string):
    if not isinstance(string, str):
        raise ValueError(f"Input {string} must be a string.")

# validate integer for dictionary function
def validate_integer(number):
    if not isinstance(number, int):
        raise ValueError("Input {number} must be an integer.")

def validate_types(data_frame, string1, string2):
    """
    Main function for the basic parameters in the px charts
    Data, X and Y axis
    Color checked in get_dict_colors, size of scatterplot is optional
    """
    validate_data_frame(data_frame)
    validate_string(string1)
    validate_string(string2)

#----------------------------------------------------------------
# color/non-plotly functions
#----------------------------------------------------------------

def get_dict_colors(df, alphabetical, color_attr, number):
    """
    You can use color OR number. If both not null, color_attr is used.
    If you use color_attr, you are using one of the attribute of the dataframe and using as many colors as many unique values color_att has.
    If you use number, you are simply giving in input how many colors you need. this is useful in cases such as line chart.

    Args:
    df: dataframe to be used to get the list of elements that need a color
    alphabetical: boolean, if true, the values are sorted alphabetically
    color_attr: string, column name, if not null, the values are sorted alphabetically
    number: int, if not null, the values are sorted alphabetically


    Example:
    color_map = get_dict_colors(df, alphabetical = False, color_attr = 'country', number = None)

    """

    # check that df is a data frame
    validate_data_frame(df)

    # if color attribute is there, check if it's a string
    # and if it's in df
    if color_attr is not None:
        validate_string(color_attr)
        validate_columns(df, [color_attr])

    # if number is there, check if it's an integer
    if number is not None:
        validate_integer(number)

    colors_mapping = {
    'Series 1': '#007EFF',
    'Series 2': '#7BD200',
    'Series 3': '#DB5700',
    'Series 4': '#FFC000',
    'Series 5': '#403152',
    'Series 6': '#BFBFBF',
    'Series 7': '#00B050',
    'Series 8': '#CC66CD',
    '1st MA': '#00379F',
    '2nd MA': '#00B0F0',
    '3rd MA': '#92CDDC'
    }

    if number is not None and color_attr is None:
        first_n_colors = list(colors_mapping.values())[:number]
        return dict(zip(list(range(number)), first_n_colors))

    unique_values = df[color_attr].unique()

    if alphabetical:
        values =  pd.Series(unique_values).sort_values().tolist()
    else:
        values =  pd.Series(unique_values).tolist()

    # if unique_values is longer than the length of the dictionary,
    # repeate dictionary colors
    if len(unique_values) > len(colors_mapping):
        first_n_colors = list(colors_mapping.values()) * (len(unique_values) // len(colors_mapping)) + list(colors_mapping.values())[:len(unique_values) % len(colors_mapping)]

        print('Warning: list of categories is longer than the amount of corporate colors, some will be repeated')

    else:
        first_n_colors = list(colors_mapping.values())[:len(values)]
    return  dict(zip(values, first_n_colors))

# Function to convert HEX to RGBA
def hex_to_rgba(hex_color, opacity=1):
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f'rgba({r}, {g}, {b}, {opacity})'



class esmaplotly:
    def __init__(self):
        # set default parameters for px.bar, px.line, px.area... functions
        self.default_title = ""

        # Define default update parameters for the update_layout function
        self.default_update_params = {
        "title_x":0.5,  # Center the title
        "title_font": dict(size=16, color='black'),  # Set title font color to black, just in case, but there should be no title
        "bargap":0.0,  # Adjust space between groups
        "bargroupgap":0.7,  # Adjust width of bars

        # leaving height and width to the user, to adjust in word?
        "height":6.28*37.795275591,  # scale in pixels vs centimeters in excel
        "width":7.57*37.795275591,  # scale in pixels vs centimeters in excel

        "plot_bgcolor":"rgba(0, 0, 0, 0)",  # Set plot background to transparent
        "paper_bgcolor":"rgba(0, 0, 0, 0)",  # Set figure background to white
        "margin":dict(b=50, t=10, l = 0, r = 0),  # Increase bottom margin for legend & annotation

        "xaxis": dict(
                tickfont=dict(size = 7, color="black"),
                tickangle = 0, # horizontally oriented labels
                nticks = 5, # User to update this if the labels are too tight or loose
                showgrid=False,  # Remove vertical gridlines
                zeroline=False  # Remove x-axis baseline
            ),

        "yaxis": dict(
            tickfont=dict(size = 7, color="black"),
            showgrid=False,  # Remove horizontal gridlines
            zeroline=False  # Remove y-axis baseline
        ),

        "legend": dict(
            font=dict(size=7,color="black"),
            title_text = '',
            orientation="h",  # Horizontal layout but wrapped
            yanchor="bottom",
            y=-0.2,  # Move legend higher so it's above annotation
            #xanchor="center",
            #x=0.5,
            traceorder="normal",
            itemwidth=30,
            itemsizing="constant",
            tracegroupgap=0,
            borderwidth = 0
        ),
        }

        # Define default annotation
        self.default_annotation = {
            "text": "Note: <br>Source: ESMA",
            "x":0.0,  # Align to the left
            "y":-0.28,  # Place it lower so it is below the legend
            "xref":"paper",
            "yref":"paper",
            "showarrow":False,  # No arrow
            "font":dict(size=6, color="black"),  # Small black font
            "align":"left",
        }

    def __getattr__(self, name):
        """
        Dynamically access attributes of plotly.express, such as scatter, bar, etc.
        """
        if hasattr(px, name):
            return getattr(px, name)
        raise AttributeError(f"'plotly.express' has no attribute '{name}'")

    def _get_params(self, x, y, **kwargs):
        """
        Generate default parameters merged with user-provided ones.

        Args:
            x: Column name for the x-axis.
            y: Column name for the y-axis.
            kwargs: Additional parameters provided by the user.

        Returns:
            Merged parameters dictionary with defaults.
        """
        default_labels = {x: "", y: ""}
        params = {
            "title": self.default_title,
            "labels": default_labels,
            **kwargs
        }
        return params


    def bar(self, data_frame, x, y, *args, **kwargs):
        """
        Custom version of px.bar with shared defaults.
        """
        params = self._get_params(x, y, **kwargs)

        # basic error handling for data_frame, x and y
        # correct types
        validate_types(data_frame, x, y)

        # columns in data_frame
        validate_columns(data_frame, [x, y])

        # create chart
        fig = px.bar(data_frame=data_frame, x=x, y=y, *args, **params)

        # border line around the bars out
        fig.update_traces(marker_line_width=0)

        return fig

    def area(self, data_frame, x, y, color, color_discrete_map, line_width = True, *args, **kwargs):
        """
        Custom version of px.area with shared defaults.
        """
        params = self._get_params(x, y, **kwargs)

        # basic error handling for data_frame, x and y
        validate_types(data_frame, x, y)

        # columns in data_frame
        validate_columns(data_frame, [x, y])

        # update color_map to have higher opacity
        # by default plotly will return more muted colours
        # in area chart than in line or bar charts
        color_discrete_map_rgba = {k: hex_to_rgba(v) for k, v in color_discrete_map.items()}

        fig = px.area(data_frame=data_frame, x=x, y=y,
                      color = color,
                      color_discrete_map = color_discrete_map_rgba,
                      *args, **params)

        # Update line width to remove border
        if line_width:
            pass
        else:
            fig.update_traces(line_width=0)

        return fig

    def line(self, data_frame, x, y, *args, **kwargs):
        """
        Custom version of px.line with shared defaults.
        """
        params = self._get_params(x, y, **kwargs)

        # basic error handling for data_frame, x and y
        validate_types(data_frame, x, y)

        # columns in data_frame
        validate_columns(data_frame, [x, y])

        # create chart
        fig = px.line(data_frame=data_frame, x=x, y=y, *args, **params)

        # No update line width to remove border, it removes the line
        #fig.update_traces(line_width=0)

        return fig

    def scatter(self, data_frame, x, y, *args, **kwargs):
        """
        Custom version of px.scatter with shared defaults.
        """
        params = self._get_params(x, y, **kwargs)

        # basic error handling for data_frame, x and y
        validate_types(data_frame, x, y)

        # columns in data_frame
        validate_columns(data_frame, [x, y])

        # create chart
        fig = px.scatter(data_frame=data_frame, x=x, y=y, *args, **params)

        return fig

    def funnel(self, data_frame, x, y, *args, **kwargs):
        """
        Custom version of px.funnel with shared defaults.
        """
        params = self._get_params(x, y, **kwargs)

        # basic error handling for data_frame, x and y
        validate_types(data_frame, x, y)

        # columns in data_frame
        validate_columns(data_frame, [x, y])

        # create chart
        fig = px.funnel(data_frame=data_frame, x=x, y=y, *args, **params)

        return fig

    def update_chart_trv(self, fig, add_annotation=True, annotation_text=None,**update_params):
        """
        Updates chart properties dynamically, with default parameters and optional annotation.

        Args:
            fig: The figure object from a plotly express method.
            add_annotation: Boolean to specify if an annotation should be added, by default 'Note: and Source:' .
            annotation_text: Custom annotation text to override the default.
            update_params: Chart parameters to update, like title_x, title_y, etc.
        """
        # Merge default layout parameters with user-provided parameters
        combined_params = {**self.default_update_params, **update_params}

        # update sizes based on scale_factor
        # kept only on the exporting side, here as reference if needed
        """
        combined_params["height"] = combined_params["height"]*scale_factor
        combined_params["width"] = combined_params["width"]*scale_factor
        combined_params['xaxis']['tickfont']['size'] = combined_params['xaxis']['tickfont']['size']*scale_factor
        combined_params['yaxis']['tickfont']['size'] = combined_params['yaxis']['tickfont']['size']*scale_factor
        combined_params['legend']['font']['size'] = combined_params['legend']['font']['size']*scale_factor
        """

        #print(combined_params)

        # update the layout properties
        fig.update_layout(**combined_params)

        # Add footnote annotation if specified
        if add_annotation:
            annotation = self.default_annotation.copy()
            if annotation_text:  # Override default text if provided
                annotation["text"] = annotation_text
            fig.update_layout(annotations=[annotation])

        return fig

    def update_layout(self, **kwargs):
        # just the plotly update_layout
        # not sure if it's needed or I was testing things wrongly
        self.fig.update_layout(**kwargs)

    def save_image(self, fig, file_name, folder_path, file_format="png", scale_factor = 1):
        """
        Write the figure to a file.

        Args:
            fig: The figure object from a plotly express method.
            file_name: The name of the file to be saved, excluding the extension.
            folder_path: The path to the folder where the file should be saved.
            format: The format of the file to be saved, such as 'png' or 'svg'.
            scale_factor: The scale factor to adjust the size of the image.
                Bigger (e.g. 5) means more pixel quality, even if shrinking later.
        """

        fig.write_image(folder_path + '/' + file_name + '.' + file_format, scale=scale_factor)
