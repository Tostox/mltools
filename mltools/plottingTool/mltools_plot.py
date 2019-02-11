from bokeh.plotting import figure
from bokeh.io import show
from bokeh.models import ColumnDataSource, Range1d, HoverTool
from bokeh.models import BasicTicker, ColorBar, LinearColorMapper, PrintfTickFormatter, LabelSet
from bokeh.models.widgets import Panel, Tabs
from bokeh.models.annotations import Span
from bokeh.transform import transform
from pandas import DataFrame
import numpy as np

# Per creare le palettes
# https://www.w3schools.com/colors/colors_mixer.asp
# http://paletton.com/#uid=74m0u0kcMdz92nwaYlkklb0nl9U


class color():
    palettes = {
        'colors_1': ['#dfccce', '#ddb7b1', '#cc7878', '#933b41', '#550b1d'],
        'colors_2': ['#f2e9e4', '#e8dfdc', '#ddd5d3', '#d3cbcb', '#c8c1c2',
                     '#beb7ba', '#b4adb1', '#a9a3a9', '#9f99a0', '#948f98',
                     '#8a8690', '#807c87', '#75727f', '#6b6876', '#605e6e',
                     '#565465', '#4c4a5d', '#414054', '#37364c', '#2c2c43',
                     '#22223b'],
        'colors_3': ['#b1f531', '#adec39', '#a9e241', '#a5d949', '#a0d051',
                     '#9cc759', '#98be61', '#94b469', '#90ab71', '#8ca279',
                     '#889882', '#838f8a', '#7f8692', '#7b7d9a', '#7774a2',
                     '#736aaa', '#6f61b2', '#6a58ba', '#664ec2', '#6245ca',
                     '#5e3cd2'],
        'corr_2': ['#d93a46', '#e98e95', '#fae6e7', '#f2f2f2', '#e9f2f5',
                   '#93b8c3', '#3f7f93'],
        'corr_3': ['#d93a46', '#da434f', '#dc4c57', '#dd5660', '#de5f68',
                   '#df6871', '#e0717a', '#e27a82', '#e3848b', '#e48d93',
                   '#e6969c', '#e79fa5', '#e8a8ad', '#e9b2b6', '#eabbbe',
                   '#ecc4c7', '#edcdd0', '#eed6d8', '#f0e0e1', '#f1e9e9',
                   '#f2f2f2', '#e9eced', '#e0e6e8', '#d7e1e4', '#cedbdf',
                   '#c5d5da', '#bccfd5', '#b3cad1', '#aac4cc', '#a1bec7',
                   '#98b8c2', '#90b3be', '#87adb9', '#7ea7b4', '#75a2b0',
                   '#6c9cab', '#6396a6', '#5a90a1', '#518a9c', '#488598',
                   '#3f7f93']
    }


class cross_validation_plot():

    def plotting_iter_res(iteration, loss, best_value=None, title=''):
        f = figure(tools="box_select, pan, reset, save")
        f.plot_width = 800
        f.plot_height = 350

        # Background settings
        f.background_fill_color = '#859dcd'
        f.background_fill_alpha = 0.05

        # Title settings
        f.title.text = title
        f.title.text_font = 'Helvetica'
        f.title.text_font_size = '24px'
        f.title.align = 'center'
        f.title.text_font_style = "italic"

        # Axis settings
        f.xaxis.axis_label = 'Iteration'
        f.yaxis.axis_label = 'Loss Value'
        f.xaxis.major_label_orientation = 0
        f.yaxis.major_label_orientation = 0
        # f.x_range = Range1d(start=min(iteration) - max(iteration) / 10, end=max(iteration) + max(iteration) / 10)
        # f.y_range = Range1d(start=min(loss) - 0.1, end=max(loss) + 0.1)

        # Grid settings
        f.xgrid.grid_line_color = None
        f.ygrid.grid_line_dash = [6, 10]

        # Plot
        f.circle(x=iteration, y=loss, size=10, fill_alpha=0.3, color='#00688f')
        if best_value:
            f.line(x=iteration, y=best_value, color='#bc4328',
                   line_dash='dashed', line_width=3, line_alpha=0.8)

        show(f)


class feature_engineering_plot():

    def _plot_features_importance(feature_importances, plot_n, threshold, figsize):

        f1 = figure(tools="box_select, pan, reset, save")
        f1.plot_width = figsize[0]
        f1.plot_height = figsize[1]

        # Background settings
        f1.background_fill_color = '#859dcd'
        f1.background_fill_alpha = 0.05

        # Title settings
        f1.title.text = 'Feature Importances'
        f1.title.text_font = 'Helvetica'
        f1.title.text_font_size = '24px'
        f1.title.align = 'center'
        f1.title.text_font_style = "italic"

        # Axis settings
        f1.xaxis.axis_label = 'Normalized Importance'
        f1.yaxis.axis_label = 'First %d Features' % plot_n
        f1.ygrid.grid_line_color = None
        f1.yaxis.ticker = feature_importances.index[:plot_n].values
        f1.axis.axis_label_text_font_size = '18px'

        source = ColumnDataSource(data=feature_importances[:plot_n][::-1])
        source.add(feature_importances[:plot_n].index, 'index')
        source.add([0.001] * plot_n, 'x_label')

        labels = LabelSet(x='x_label', y='index', text='feature',
                          x_offset=0, y_offset=-8, text_font_size='14px', text_color='black',
                          level='glyph', source=source, render_mode='canvas')

        # Need to reverse the index to plot most important on top
        f1.hbar(right='normalized_importance', y='index', source=source,
                color='#3399c1', alpha=0.7, height=0.9)

        f1.add_layout(labels)

        hover = HoverTool(tooltips=[('Importance', '@importance{0.0000}')])
        f1.add_tools(hover)

        tab1 = Panel(child=f1, title='Features Importance')

        # Cumulative importance plot
        f2 = figure(tools="box_select, box_zoom, pan, reset, save")
        f2.plot_width = figsize[0]
        f2.plot_height = figsize[1]

        # Background settings
        f2.background_fill_color = '#859dcd'
        f2.background_fill_alpha = 0.05

        # Title settings
        f2.title.text = 'Cumulative Feature Importance'
        f2.title.text_font = 'Helvetica'
        f2.title.text_font_size = '24px'
        f2.title.align = 'center'
        f2.title.text_font_style = "italic"

        # Axis settings
        f2.xaxis.axis_label = 'Number of Features'
        f2.yaxis.axis_label = 'Cumulative Importance'
        f2.xgrid.grid_line_color = None
        f2.axis.axis_label_text_font_size = '18px'

        source = ColumnDataSource(data=feature_importances)
        source.add(list(range(1, len(feature_importances) + 1)), 'index')

        f2.line(x='index', y='cumulative_importance', source=source,
                color='#3399c1')

        if threshold:
            # Index of minimum number of features needed for cumulative importance threshold
            # np.where returns the index so need to add 1 to have correct number
            importance_index = np.min(np.where(feature_importances['cumulative_importance'] > threshold))
            span = Span(location=importance_index,
                        dimension='height', line_color='red',
                        line_dash='dashed', line_width=2)
            f2.add_layout(span)
            # threshold_source = ColumnDataSource(data={'importance_index': [importance_index + 1] * 2,
            #                                           'importance_value': [0, 1]})

            # f2.line(x='importance_index', y='importance_value', source=threshold_source,
            #         line_dash=[4, 2], line_color='red')

        hover = HoverTool(tooltips=[('Feature', '@feature'),
                                    ('Feature Importance', '@importance{0.0000}'),
                                    ('Cumulative Importance', '@cumulative_importance')],
                          mode='hline')
        f2.add_tools(hover)

        tab2 = Panel(child=f2, title='Cumulative Importance')
        tabs = Tabs(tabs=[tab1, tab2])

        show(tabs)

        print('%d features required for %0.2f of cumulative importance' % (importance_index + 1, threshold))

    def _plot_unique_value(data, figsize):

        f = figure(x_range=list(data.index), tools="box_select, pan, reset, save")
        f.plot_width = figsize[0]
        f.plot_height = figsize[1]

        # Background settings
        f.background_fill_color = '#859dcd'
        f.background_fill_alpha = 0.05

        # Title settings
        f.title.text = 'Number of Unique Values Histogram'
        f.title.text_font = 'Helvetica'
        f.title.text_font_size = '24px'
        f.title.align = 'center'
        f.title.text_font_style = "italic"

        # Axis settings
        f.xaxis.axis_label = 'Unique Values'
        f.xaxis.major_label_orientation = np.pi / 3
        f.yaxis.axis_label = 'Frequency'
        f.axis.axis_label_text_font_size = '16px'

        f.xgrid.grid_line_color = None

        source = ColumnDataSource()
        source.add(data.index, 'features')
        source.add(data['nunique'], 'nunique')

        f.vbar(x='features', top='nunique', source=source,
               alpha=0.7, width=0.9, color='#3399c1')

        hover = HoverTool(tooltips=[('Feature', '@features'),
                                    ('# Values', '@nunique')])
        f.add_tools(hover)

        show(f)

    def _plot_collinear(corr_matrix_plot, title, figsize):

        df = DataFrame(corr_matrix_plot.stack(), columns=['corr']).reset_index()
        source = ColumnDataSource(df)

        f = figure(x_range=list(corr_matrix_plot.index),
                   y_range=list(reversed(list(corr_matrix_plot.columns))),
                   tools="box_select, pan, reset, save")
        f.plot_width = figsize[0]
        f.plot_height = figsize[1]

        # Title settings
        f.title.text = title
        f.title.text_font = 'Helvetica'
        f.title.text_font_size = '24px'
        f.title.align = 'center'
        f.title.text_font_style = "italic"

        # Axis settings
        f.axis.axis_label_text_font = 'Helvetica'
        f.axis.axis_label_text_font_size = '18px'
        f.xaxis.major_label_orientation = np.pi / 2

        mapper = LinearColorMapper(palette=color.palettes['corr_3'][::-1], low=-1, high=1)

        f.rect(x='level_0', y='level_1', width=1, height=1, source=source,
               line_color=None, fill_color=transform('corr', mapper))

        color_bar = ColorBar(color_mapper=mapper, location=(0, 0),
                             ticker=BasicTicker(desired_num_ticks=len(color.palettes['corr_3'])),
                             formatter=PrintfTickFormatter(format="%.2f"))

        f.add_layout(color_bar, 'right')

        hover = HoverTool(tooltips=[("Correlations", "@corr"),
                                    ("Feature 1", "@level_0"),
                                    ("Feature 2", "@level_1")])
        f.add_tools(hover)

        show(f)


class time_series_plot():

    def plot_time_series(data, fig, fig_size=[800, 350], title="", legend="", labels=["Date", ""],
                         linewidth=1, color="blue", linestyle="solid", alpha=1):

        fig.plot_width = fig_size[0]
        fig.plot_height = fig_size[1]

        source = ColumnDataSource()
        source.add(data.index, name="date")
        source.add(data.values, name="values")
        source.add(data.index.strftime("%Y-%m-%d %H:%M:%S"), "date_formatted")

        # create a line plot
        fig.line(source=source, x="date", y="values",
                 legend=legend, line_color=color, line_dash=linestyle, line_alpha=alpha, line_width=linewidth)

        fig.title.text = title
        fig.title.text_font = 'Helvetica'
        fig.title.text_font_size = '18px'
        fig.title.align = 'center'

        fig.legend.background_fill_alpha = 0.6
        fig.legend.label_text_font = 'Helvetica'
        fig.legend.location = 'bottom_right'

        fig.xaxis.axis_label = labels[0]
        fig.yaxis.axis_label = labels[1]

        hover = HoverTool(tooltips=[("Value: ", "@values"), ("Timestamp: ", "@date_formatted")])
        hover.formatters = {'Timestamp': "datetime"}
        fig.add_tools(hover)
        fig.toolbar_location = "above"

        return fig

    def _plot_diagnostics(residuals):

        f = figure(tools="box_select, pan, reset, save")
        f.plot_width = 300
        f.plot_height = 300

        # Background settings
        f.background_fill_color = '#859dcd'
        f.background_fill_alpha = 0.05

        # Axis settings
        f.axis.axis_label_text_font = 'Helvetica'

        # Title settings
        f.title.text_font = 'Helvetica'
        f.title.text_font_size = '18px'
        f.title.align = 'center'

        f1 = f
        f1.title.text = "Standardized Residuals"
        f1.line(residuals, line_color='blue', linewidth=1)
        f1.line(residuals * 0, line_color='red', linewidth=1)


        sns.distplot(residuals, bins=bins, hist=True, kde=True, color='blue',
                     hist_kws={'color': 'blue', 'label': 'Hist'},
                     kde_kws={'linewidth': 2, 'label': 'KDE'}, ax=axes[0, 1])

        value = np.random.normal(loc=0, scale=1, size=10000000)
        sns.distplot(value, hist=False, ax=axes[0, 1], color='red', label='N(0,1)')
        axes[0, 1].set_title('Residuals Histogram and Density', fontsize='large')

        # # pd_series is the series you want to plot
        # series1 = probplot(pd_series, dist="norm")
        # p1 = figure(title="Normal QQ-Plot", background_fill_color="#E8DDCB")
        # p1.scatter(series1[0][0], series1[0][1], fill_color="red")
        # show(p1)

        stats.probplot(residuals, plot=axes[1, 0])

        plot_acf(residuals, ax=axes[1, 1], lags=lags, alpha=alpha,
                 vlines_kwargs={'color': 'darkblue'})

        plt.show()


    def _plot_wcorr(Wcorr, L):

        f = figure(tools="box_select, pan, reset, save")
        f.plot_width = 700
        f.plot_height = 600

        # Background settings
        f.background_fill_color = '#859dcd'
        f.background_fill_alpha = 0.05

        # Title settings
        f.title.text = "W-Correlation for L={}".format(L)
        f.title.text_font = 'Helvetica'
        f.title.text_font_size = '24px'
        f.title.align = 'center'
        f.title.text_font_style = "italic"

        # Axis settings
        f.xaxis.axis_label = 'Fⱼ'
        f.yaxis.axis_label = 'Fᵢ'
        f.axis.axis_label_text_font = 'Helvetica'
        f.axis.axis_label_text_font_size = '24px'
        f.axis.major_label_orientation = 0
        f.x_range = Range1d(start=0.5, end=L + 0.5)
        f.y_range = Range1d(start=L + 0.5, end=0.5)
        f.axis[0].ticker.desired_num_ticks = L
        f.axis[0].ticker.num_minor_ticks = 0

        data = DataFrame(Wcorr)
        axis = [i for i in range(1, Wcorr.shape[0] + 1)]

        data['F_i'] = axis
        data.set_index('F_i', inplace=True)

        data.columns = axis
        data.columns.name = 'F_j'
        df = DataFrame(data.stack(), columns=['corr']).reset_index()
        source = ColumnDataSource(df)

        # this is the colormap from the original NYTimes plot

        mapper = LinearColorMapper(palette=color.palettes['colors_2'], low=0, high=1)

        f.rect(x="F_i", y="F_j", width=1, height=1, source=source,
               line_color=None, fill_color=transform('corr', mapper))

        color_bar = ColorBar(color_mapper=mapper, location=(0, 0),
                             ticker=BasicTicker(desired_num_ticks=len(color.palettes['colors_2'])),
                             formatter=PrintfTickFormatter(format="%.2f"))

        f.add_layout(color_bar, 'right')

        hover = HoverTool(tooltips=[("Components", "(@F_i, @F_j)"),
                                    ("Correlations", "@corr")])
        f.add_tools(hover)

        show(f)


class text_mining_plot():

    def _plot_important_words(importance, classes, figsize):
        tab_list = []

        for class_name in classes:

            top_scores = [a[0] for a in importance[class_name]['tops']]
            top_words = [a[1] for a in importance[class_name]['tops']]
            top_pairs = [(a, b) for a, b in zip(top_words, top_scores)]
            top_pairs = sorted(top_pairs, key=lambda x: x[1])
            top_words = [a[0] for a in top_pairs]
            top_scores = [a[1] for a in top_pairs]

            # Figure
            f = figure(tools="box_select, pan, reset, save")
            f.plot_width = figsize[0]
            f.plot_height = figsize[1]

            # Background settings
            f.background_fill_color = '#859dcd'
            f.background_fill_alpha = 0.05

            # Title settings
            f.title.text = 'Feature Importance for "%s"' % class_name
            f.title.text_font = 'Helvetica'
            f.title.text_font_size = '24px'
            f.title.align = 'center'
            f.title.text_font_style = "italic"

            # Axis settings
            f.xaxis.axis_label = 'Importance'
            f.yaxis.axis_label = 'Most important words'
            f.axis.axis_label_text_font_size = '16px'

            # Grid settings
            f.xgrid.grid_line_color = None

            # Create source
            df = DataFrame({'top_words': top_words, 'top_scores': top_scores})
            source = ColumnDataSource(data=df)
            source.add(df.index, 'index')
            source.add([0.2] * df.shape[0], 'x_label')

            labels = LabelSet(x='x_label', y='index', text='top_words',
                              x_offset=0, y_offset=-8, text_font_size='16px', text_color='black',
                              level='glyph', source=source, render_mode='canvas')

            # Plot
            f.hbar(y='index', right='top_scores', source=source, height=0.9,
                   alpha=0.7, color='#3399c1')

            f.add_layout(labels)

            hover = HoverTool(tooltips=[("Importance", "@top_scores")])
            f.add_tools(hover)


            tab = Panel(child=f, title=class_name)
            tab_list.append(tab)

        tabs = Tabs(tabs=tab_list)
        show(tabs)

    def _plot_word_freq(frequencies_dist, plot_n, figsize):
        tab_list = []
        for label in frequencies_dist.keys():

            freqdist = frequencies_dist[label]
            df = DataFrame({'word': list(freqdist.keys()),
                            'count': list(freqdist.values())})
            df.sort_values('count', inplace=True, ascending=False)
            df.reset_index(inplace=True, drop=True)

            source = ColumnDataSource(data=df[:plot_n])
            source.add(df.index[:plot_n], 'index')

            f = figure(x_range=list(df['word'])[:plot_n],
                       tools="box_select, pan, reset, save")
            f.plot_width = figsize[0]
            f.plot_height = figsize[1]

            f.title.text = 'Word frequency for "%s"' % label
            f.title.text_font = 'Helvetica'
            f.title.text_font_size = '24px'
            f.title.align = 'center'

            f.xaxis.axis_label = 'Samples'
            f.yaxis.axis_label = 'Count'
            f.xaxis.major_label_orientation = np.pi / 2

            f.line(source=source, x='word', y='count')

            hover = HoverTool(tooltips=[("Word", "@word"), ("Count", "@count")], mode='vline')
            f.add_tools(hover)

            tab = Panel(child=f, title=label)
            tab_list.append(tab)

        tabs = Tabs(tabs=tab_list)
        show(tabs)
