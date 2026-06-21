"""Time series plot renderer for DAVINCI.

This module provides time series plotting functionality for comparing
dataset output with datasets over time.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from davinci_monet.plots import labeling
from davinci_monet.plots.base import (
    BasePlotter,
    get_axis_color,
    get_variable_label,
    get_variable_units,
    series_colors,
)
from davinci_monet.plots.registry import register_plotter

if TYPE_CHECKING:
    import matplotlib.axes
    import matplotlib.figure
    import xarray as xr

    from davinci_monet.core.base import PlotSeries


@register_plotter("timeseries")
class TimeSeriesPlotter(BasePlotter):
    """Plotter for time series comparisons.

    Creates line plots showing x and y values over time.
    Supports temporal resampling and uncertainty bands.

    Parameters
    ----------
    config
        Plot configuration.

    Examples
    --------
    >>> plotter = TimeSeriesPlotter()
    >>> fig = plotter.render(
    ...     build_series(paired_data, "x_o3", "y_o3"),
    ...     resample="1h",
    ... )
    """

    name: str = "timeseries"
    default_figsize: tuple[float, float] = (9, 4)  # Wide for temporal data

    def _plot(
        self,
        paired_data: xr.Dataset,
        x_var: str,
        y_var: str,
        ax: matplotlib.axes.Axes | None = None,
        resample: str | None = None,
        show_uncertainty: bool = False,
        uncertainty_type: Literal["std", "iqr", "range"] = "std",
        time_dim: str = "time",
        aggregate_dim: str | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        show_individual_sites: bool = False,
        site_dim: str = "site",
        site_label_var: str = "site_name",
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Generate a time series plot.

        Parameters
        ----------
        paired_data
            Paired dataset with x and y variables.
        x_var
            Name of the x variable.
        y_var
            Name of the y variable.
        ax
            Optional axes to plot on. If None, creates new figure.
        resample
            Temporal resampling frequency (e.g., '1h', '1D').
        show_uncertainty
            If True, show uncertainty bands around mean.
        uncertainty_type
            Type of uncertainty band ('std', 'iqr', 'range').
        time_dim
            Name of time dimension.
        aggregate_dim
            Optional dimension to aggregate over (e.g., 'site').
        x_label
            Custom label for the x series.
        y_label
            Custom label for the y series.
        **kwargs
            Additional plotting arguments.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        # Create figure if needed
        if ax is None:
            fig, ax = self.create_figure()
        else:
            fig = ax.get_figure()  # type: ignore[assignment]

        # Get data arrays
        x_data = paired_data[x_var]
        y_data = paired_data[y_var]

        # Plot individual sites if requested
        if show_individual_sites and site_dim in x_data.dims:
            return self._plot_individual_sites(
                fig,
                ax,
                paired_data,
                x_var,
                y_var,
                time_dim,
                site_dim,
                site_label_var,
                resample,
                **kwargs,
            )

        # Aggregate over non-time dimensions if specified
        if aggregate_dim is not None and aggregate_dim in x_data.dims:
            x_data = x_data.mean(dim=aggregate_dim)
            y_data = y_data.mean(dim=aggregate_dim)
        elif len(x_data.sizes) > 1:
            # If multiple dimensions and no aggregate specified, average over all except time
            other_dims = [d for d in x_data.sizes if d != time_dim]
            if other_dims:
                x_data = x_data.mean(dim=other_dims)
                y_data = y_data.mean(dim=other_dims)

        # Get time coordinate
        time = paired_data[time_dim]

        # Resample if requested
        if resample:
            x_data = x_data.resample({time_dim: resample}).mean()
            y_data = y_data.resample({time_dim: resample}).mean()
            time = x_data[time_dim]

        # Convert to numpy for plotting
        time_values = pd.to_datetime(time.values)
        x_values = x_data.values
        y_values = y_data.values

        # Get style configuration
        style = self.config.style

        # Series legend labels prefer source identity (e.g. airnow/cam); axis
        # remains a styling hint only.  Route through labeling.legend_label so
        # the displayed name is always friendly (never a raw config key).
        def _series_legend(var: str, config_label: str | None) -> str:
            if config_label:
                return config_label
            src = paired_data[var].attrs.get("source_label") if var in paired_data else None
            if src:
                return labeling.legend_label(src)
            return get_variable_label(paired_data, var, include_prefix=False)

        x_label = x_label or _series_legend(x_var, self.config.x_label)
        y_label = y_label or _series_legend(y_var, self.config.y_label)

        # Series colors by source axis (x gray, y blue, else palette); a
        # customised StyleConfig still wins for the x/y axes (R-3).
        x_color = get_axis_color(
            paired_data,
            x_var,
            0,
            x_color=style.x_color,
            y_color=style.y_color,
        )
        y_color = get_axis_color(
            paired_data,
            y_var,
            1,
            x_color=style.x_color,
            y_color=style.y_color,
        )

        # Plot datasets
        ax.plot(
            time_values,
            x_values,
            color=x_color,
            linestyle=style.x_linestyle,
            marker=style.x_marker if len(time_values) < 50 else None,
            linewidth=style.linewidth,
            markersize=style.markersize,
            alpha=style.alpha,
            label=x_label,
        )

        # Plot dataset
        ax.plot(
            time_values,
            y_values,
            color=y_color,
            linestyle=style.y_linestyle,
            marker=style.y_marker if len(time_values) < 50 else None,
            linewidth=style.linewidth,
            markersize=style.markersize,
            alpha=style.alpha,
            label=y_label,
        )

        # Show uncertainty bands if requested (requires ungrouped data)
        if show_uncertainty and aggregate_dim is not None:
            self._add_uncertainty_bands(
                ax,
                paired_data,
                x_var,
                y_var,
                time_dim,
                aggregate_dim,
                resample,
                uncertainty_type,
            )

        # Set x-axis limits to actual data range (avoid extra ticks beyond data)
        ax.set_xlim(time_values.min(), time_values.max())

        # Formatting
        self.apply_text_style(ax)

        # Set labels - use automatic variable display name (no prefix for shared axis).
        # Always include SI-formatted units via labeling.axis_label so units are
        # never silently omitted.
        units = get_variable_units(paired_data, x_var)
        quantity = self.config.ylabel or get_variable_label(
            paired_data, x_var, include_prefix=False
        )
        ylabel = labeling.axis_label(quantity, units)
        self.set_labels(ax, xlabel="Time", ylabel=ylabel)

        # Smart auto-scaling for y-axis
        self._set_smart_ylim(
            ax,
            paired_data,
            x_var,
            y_var,
            aggregate_dim,
            time_dim,
            resample,
            show_uncertainty,
            uncertainty_type,
        )

        # Add legend
        self.add_legend(ax)

        # Rotate x-axis labels for readability
        ax.tick_params(axis="x", rotation=45)

        # Grid
        ax.grid(True, alpha=0.3)

        return fig

    def render(
        self,
        series: list[PlotSeries],
        ax: matplotlib.axes.Axes | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Unified renderer entry: render 1..N source series.

        - ``1`` series → a single line, aggregated over non-time dims by default
          (the cross-site mean — no more one-line-per-site spaghetti). Opt into
          per-site lines with ``show_individual_sites=True`` and a ±1σ band with
          ``show_uncertainty=True``.
        - ``2`` series → x-vs-y; delegates to the paired implementation so
          x-gray/y-blue styling is unchanged.
        - ``>2`` series → multi-source overlay, palette-cycled.
        """
        if len(series) == 2:
            x_series = next((s for s in series if s.axis == "x"), series[0])
            y_series = next((s for s in series if s.axis == "y"), series[1])
            return self._plot(
                x_series.dataset,
                x_series.var_name,
                y_series.var_name,
                ax=ax,
                **kwargs,
            )
        if len(series) == 1:
            return self._render_single(series[0], ax=ax, **kwargs)
        return self._render_overlay(series, ax=ax, **kwargs)

    def _render_single(
        self,
        s: PlotSeries,
        ax: matplotlib.axes.Axes | None = None,
        *,
        title: str | None = None,
        color: str | None = None,
        show_altitude: bool = False,
        alt_coord: str = "altitude",
        show_uncertainty: bool = False,
        uncertainty_type: Literal["std", "iqr", "range"] = "std",
        show_individual_sites: bool = False,
        time_dim: str = "time",
        aggregate_dim: str | None = None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Render one source series as a single (aggregated) line.

        ``show_uncertainty`` shades a band about the mean across the aggregated
        dimension(s); ``uncertainty_type`` selects ``std`` (mean ± 1σ), ``iqr``
        (Q1–Q3), or ``range`` (min–max).
        """
        if ax is None:
            fig, ax = self.create_figure()
        else:
            fig = ax.get_figure()  # type: ignore[assignment]

        ds = s.dataset
        da = ds[s.var_name]
        color = color or series_colors([s])[0]
        if s.source_label:
            label = labeling.legend_label(s.source_label)
        else:
            label = get_variable_label(ds, s.var_name, include_prefix=False)
        time_values = pd.to_datetime(ds[time_dim].values)
        non_time_dims = [d for d in da.dims if d != time_dim]

        if show_individual_sites and non_time_dims:
            import matplotlib.cm as cm

            sdim = aggregate_dim if (aggregate_dim in non_time_dims) else non_time_dims[0]
            n = ds.sizes[sdim]
            palette = cm.tab20(np.linspace(0, 1, min(n, 20)))  # type: ignore[attr-defined]
            for i in range(n):
                ax.plot(
                    time_values,
                    da.isel({sdim: i}).values,
                    color=palette[i % len(palette)],
                    linewidth=1.0,
                    alpha=0.7,
                )
        else:
            agg_dims = (
                [aggregate_dim] if (aggregate_dim and aggregate_dim in da.dims) else non_time_dims
            )
            mean = da.mean(dim=agg_dims) if agg_dims else da
            ax.plot(time_values, mean.values, color=color, linewidth=1.5, label=label)
            if show_uncertainty and agg_dims:
                if uncertainty_type == "iqr":
                    lower = da.quantile(0.25, dim=agg_dims)
                    upper = da.quantile(0.75, dim=agg_dims)
                elif uncertainty_type == "range":
                    lower = da.min(dim=agg_dims)
                    upper = da.max(dim=agg_dims)
                else:  # "std": ±1σ about the mean
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", "Degrees of freedom", RuntimeWarning)
                        std = da.std(dim=agg_dims)
                    lower, upper = mean - std, mean + std
                ax.fill_between(
                    time_values,
                    lower.values,
                    upper.values,
                    color=color,
                    alpha=0.25,
                    linewidth=0,
                )

        # Exact date range: pin the x-axis to the data extent (no auto-padding),
        # matching the paired _plot() path.
        if len(time_values):
            ax.set_xlim(time_values.min(), time_values.max())

        units = ds[s.var_name].attrs.get("units") or None
        var_label = get_variable_label(ds, s.var_name, include_prefix=False)
        ax.set_ylabel(labeling.axis_label(var_label, units), fontsize=self.config.text.fontsize)
        ax.set_xlabel("Time", fontsize=self.config.text.fontsize)
        # Synthesise a title when none is configured (mirrors the spatial
        # renderer): source name first, then quantity, so a single-source series
        # shows e.g. "AirNow O3" rather than only the date subtitle.
        if title or self.config.title:
            self.set_title(ax, title)
        else:
            title_q = labeling.quantity_label(ds, s.var_name)
            self.set_title(ax, labeling.sourced_title(s.source_label, title_q))
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=45)

        if show_altitude and alt_coord in ds.coords:
            ax2 = ax.twinx()
            ax2.plot(  # type: ignore[attr-defined]
                time_values, ds[alt_coord].values, color="#AAAAAA", linewidth=0.8, alpha=0.6
            )
            alt_units = ds[alt_coord].attrs.get("units", "")
            ax2.set_ylabel(
                f"Altitude ({alt_units})" if alt_units else "Altitude",
                fontsize=self.config.text.fontsize,
                color="#AAAAAA",
            )
            ax2.tick_params(axis="y", labelcolor="#AAAAAA")
        return fig

    def _render_overlay(
        self,
        series: list[PlotSeries],
        ax: matplotlib.axes.Axes | None = None,
        *,
        title: str | None = None,
        time_dim: str = "time",
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Render N>2 source series as a palette-cycled overlay of mean lines."""
        if ax is None:
            fig, ax = self.create_figure()
        else:
            fig = ax.get_figure()  # type: ignore[assignment]

        colors = series_colors(series)
        for s, c in zip(series, colors):
            da = s.dataset[s.var_name]
            non_time_dims = [d for d in da.dims if d != time_dim]
            mean = da.mean(dim=non_time_dims) if non_time_dims else da
            if s.source_label:
                label = labeling.legend_label(s.source_label)
            else:
                label = get_variable_label(s.dataset, s.var_name, include_prefix=False)
            ax.plot(
                pd.to_datetime(s.dataset[time_dim].values),
                mean.values,
                color=c,
                linewidth=1.5,
                label=label,
            )
        ax.legend(fontsize=self.config.text.legend)

        # Exact date range across all overlaid series (no auto-padding).
        all_times = [pd.to_datetime(s.dataset[time_dim].values) for s in series]
        all_times = [t for t in all_times if len(t)]
        if all_times:
            ax.set_xlim(min(t.min() for t in all_times), max(t.max() for t in all_times))

        first = series[0]
        units = first.dataset[first.var_name].attrs.get("units") or None
        var_label = get_variable_label(first.dataset, first.var_name, include_prefix=False)
        ax.set_ylabel(labeling.axis_label(var_label, units), fontsize=self.config.text.fontsize)
        ax.set_xlabel("Time", fontsize=self.config.text.fontsize)
        # Overlay sources are distinguished by the legend, so the synthesised
        # title carries the quantity only (no single source name).
        if title or self.config.title:
            self.set_title(ax, title)
        else:
            self.set_title(ax, labeling.quantity_label(first.dataset, first.var_name))
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=45)
        return fig

    def _plot_individual_sites(
        self,
        fig: matplotlib.figure.Figure,
        ax: matplotlib.axes.Axes,
        paired_data: xr.Dataset,
        x_var: str,
        y_var: str,
        time_dim: str,
        site_dim: str,
        site_label_var: str,
        resample: str | None,
        **kwargs: Any,
    ) -> matplotlib.figure.Figure:
        """Plot individual site time series.

        Parameters
        ----------
        fig
            Figure object.
        ax
            Axes to plot on.
        paired_data
            Paired dataset.
        x_var, y_var
            Variable names.
        time_dim
            Time dimension name.
        site_dim
            Site dimension name.
        site_label_var
            Variable containing site labels.
        resample
            Resampling frequency.

        Returns
        -------
        matplotlib.figure.Figure
            The generated figure.
        """
        import matplotlib.cm as cm

        x_data = paired_data[x_var]
        y_data = paired_data[y_var]
        time_values = pd.to_datetime(paired_data[time_dim].values)

        # Get site labels
        if site_label_var in paired_data.coords:
            site_labels = paired_data[site_label_var].values
        else:
            site_labels = [f"Site {i}" for i in range(paired_data.sizes[site_dim])]  # type: ignore[assignment]

        n_sites = paired_data.sizes[site_dim]

        # Use a colormap for different sites
        colors = cm.tab20(np.linspace(0, 1, min(n_sites, 20)))  # type: ignore[attr-defined]

        # Plot each site
        for i in range(n_sites):
            site_x = x_data.isel({site_dim: i})
            site_y = y_data.isel({site_dim: i})

            # Skip if all NaN
            if site_x.isnull().all() and site_y.isnull().all():
                continue

            color = colors[i % len(colors)]
            label = str(site_labels[i]) if i < len(site_labels) else f"Site {i}"

            # Plot x series as solid lines
            ax.plot(
                time_values,
                site_x.values,
                color=color,
                linestyle="-",
                marker="o",
                markersize=4,
                linewidth=1,
                alpha=0.7,
                label=label,
            )

            # Plot y series as dashed lines (same site colour, dashed style
            # is the visual discriminator; suppress duplicate legend entry)
            ax.plot(
                time_values,
                site_y.values,
                color=color,
                linestyle="--",
                marker="s",
                markersize=4,
                linewidth=1,
                alpha=0.7,
                label="_" + label,
            )

        # Formatting
        self.apply_text_style(ax)

        # Set labels - use automatic variable display name (no prefix for shared axis).
        # Always include SI-formatted units via labeling.axis_label.
        units = get_variable_units(paired_data, x_var)
        quantity = self.config.ylabel or get_variable_label(
            paired_data, x_var, include_prefix=False
        )
        ylabel = labeling.axis_label(quantity, units)
        self.set_labels(ax, xlabel="Time", ylabel=ylabel)

        # Legend - put outside plot if many sites
        if n_sites > 5:
            ax.legend(
                bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=self.config.text.legend_small
            )
            fig.tight_layout()
        else:
            ax.legend(fontsize=self.config.text.legend)

        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.3)

        return fig

    def _add_uncertainty_bands(
        self,
        ax: matplotlib.axes.Axes,
        paired_data: xr.Dataset,
        x_var: str,
        y_var: str,
        time_dim: str,
        aggregate_dim: str,
        resample: str | None,
        uncertainty_type: str,
    ) -> None:
        """Add uncertainty bands to the plot.

        Parameters
        ----------
        ax
            Axes to add bands to.
        paired_data
            Full paired dataset.
        x_var, y_var
            Variable names.
        time_dim
            Time dimension name.
        aggregate_dim
            Dimension being aggregated.
        resample
            Resampling frequency.
        uncertainty_type
            Type of uncertainty ('std', 'iqr', 'range').
        """
        x_data = paired_data[x_var]
        y_data = paired_data[y_var]

        # Resample first if needed
        if resample:
            x_data = x_data.resample({time_dim: resample}).mean()
            y_data = y_data.resample({time_dim: resample}).mean()

        time_values = pd.to_datetime(x_data[time_dim].values)
        style = self.config.style

        # Calculate uncertainty bounds
        if uncertainty_type == "std":
            x_mean = x_data.mean(dim=aggregate_dim)
            y_mean = y_data.mean(dim=aggregate_dim)

            # Suppress warnings for time bins with single datasets (ddof > n)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", "Degrees of freedom", RuntimeWarning)
                x_std = x_data.std(dim=aggregate_dim)
                y_std = y_data.std(dim=aggregate_dim)

            x_lower = x_mean - x_std
            x_upper = x_mean + x_std
            y_lower = y_mean - y_std
            y_upper = y_mean + y_std

        elif uncertainty_type == "iqr":
            x_lower = x_data.quantile(0.25, dim=aggregate_dim)
            x_upper = x_data.quantile(0.75, dim=aggregate_dim)
            y_lower = y_data.quantile(0.25, dim=aggregate_dim)
            y_upper = y_data.quantile(0.75, dim=aggregate_dim)

        else:  # range
            x_lower = x_data.min(dim=aggregate_dim)
            x_upper = x_data.max(dim=aggregate_dim)
            y_lower = y_data.min(dim=aggregate_dim)
            y_upper = y_data.max(dim=aggregate_dim)

        # Plot bands (pair-axis colors, matching the series; R-3)
        ax.fill_between(
            time_values,
            x_lower.values,
            x_upper.values,
            color=get_axis_color(
                paired_data,
                x_var,
                0,
                x_color=style.x_color,
                y_color=style.y_color,
            ),
            alpha=0.2,
        )
        ax.fill_between(
            time_values,
            y_lower.values,
            y_upper.values,
            color=get_axis_color(
                paired_data,
                y_var,
                1,
                x_color=style.x_color,
                y_color=style.y_color,
            ),
            alpha=0.2,
        )

    def _set_smart_ylim(
        self,
        ax: matplotlib.axes.Axes,
        paired_data: xr.Dataset,
        x_var: str,
        y_var: str,
        aggregate_dim: str | None,
        time_dim: str,
        resample: str | None,
        show_uncertainty: bool,
        uncertainty_type: str,
    ) -> None:
        """Set smart y-axis limits based on data range.

        Computes appropriate limits from the actual data, including
        uncertainty bands if shown. Uses vmin=0 for non-negative data
        and adds padding to vmax.

        Parameters
        ----------
        ax
            Axes to configure.
        paired_data
            Full paired dataset.
        x_var, y_var
            Variable names.
        aggregate_dim
            Dimension being aggregated.
        time_dim
            Time dimension name.
        resample
            Resampling frequency.
        show_uncertainty
            Whether uncertainty bands are shown.
        uncertainty_type
            Type of uncertainty ('std', 'iqr', 'range').
        """
        # If config specifies both limits, use them
        if self.config.vmin is not None and self.config.vmax is not None:
            ax.set_ylim(self.config.vmin, self.config.vmax)
            return

        # Get data for computing range
        x_data = paired_data[x_var]
        y_data = paired_data[y_var]

        # Resample if needed
        if resample:
            x_data = x_data.resample({time_dim: resample}).mean()
            y_data = y_data.resample({time_dim: resample}).mean()

        # Compute the data range we need to display
        if show_uncertainty and aggregate_dim is not None:
            # Need to include uncertainty bands in range calculation
            if uncertainty_type == "std":
                x_mean = x_data.mean(dim=aggregate_dim)
                y_mean = y_data.mean(dim=aggregate_dim)

                # Suppress warnings for time bins with single datasets (ddof > n)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", "Degrees of freedom", RuntimeWarning)
                    x_std = x_data.std(dim=aggregate_dim)
                    y_std = y_data.std(dim=aggregate_dim)

                data_min = float(
                    min(
                        np.nanmin(x_mean.values - x_std.values),
                        np.nanmin(y_mean.values - y_std.values),
                    )
                )
                data_max = float(
                    max(
                        np.nanmax(x_mean.values + x_std.values),
                        np.nanmax(y_mean.values + y_std.values),
                    )
                )
            elif uncertainty_type == "iqr":
                data_min = float(
                    min(
                        np.nanmin(x_data.quantile(0.25, dim=aggregate_dim).values),
                        np.nanmin(y_data.quantile(0.25, dim=aggregate_dim).values),
                    )
                )
                data_max = float(
                    max(
                        np.nanmax(x_data.quantile(0.75, dim=aggregate_dim).values),
                        np.nanmax(y_data.quantile(0.75, dim=aggregate_dim).values),
                    )
                )
            else:  # range
                data_min = float(
                    min(
                        np.nanmin(x_data.min(dim=aggregate_dim).values),
                        np.nanmin(y_data.min(dim=aggregate_dim).values),
                    )
                )
                data_max = float(
                    max(
                        np.nanmax(x_data.max(dim=aggregate_dim).values),
                        np.nanmax(y_data.max(dim=aggregate_dim).values),
                    )
                )
        else:
            # Just use mean values. Replicate _plot()'s aggregation so that
            # ylim is computed from the actual plotted line, not the raw
            # per-site/per-track-point distribution. Without this, a single
            # outlier (e.g. one wildfire-impacted PM2.5 site at 200 µg/m³)
            # drives vmax far above the cross-site mean that's plotted.
            if aggregate_dim is not None and aggregate_dim in x_data.dims:
                x_data = x_data.mean(dim=aggregate_dim)
                y_data = y_data.mean(dim=aggregate_dim)
            else:
                other_dims = [d for d in x_data.sizes if d != time_dim]
                if other_dims:
                    x_data = x_data.mean(dim=other_dims)
                    y_data = y_data.mean(dim=other_dims)

            data_min = float(min(np.nanmin(x_data.values), np.nanmin(y_data.values)))
            data_max = float(max(np.nanmax(x_data.values), np.nanmax(y_data.values)))

        # Check if raw data is non-negative (physical constraint)
        # Use original data before aggregation to check this
        raw_x = paired_data[x_var]
        raw_y = paired_data[y_var]
        raw_min = float(min(np.nanmin(raw_x.values), np.nanmin(raw_y.values)))
        is_positive_definite = raw_min >= 0

        # Add padding (10% of range)
        data_range = data_max - data_min
        padding = data_range * 0.1 if data_range > 0 else 0.1

        # Determine vmin: use 0 for non-negative data, otherwise add padding
        if self.config.vmin is not None:
            vmin = self.config.vmin
        elif is_positive_definite:
            # Non-negative data (concentrations, AOD, etc.) - start at 0
            vmin = 0.0
        else:
            vmin = data_min - padding

        # Determine vmax: use config or add padding
        if self.config.vmax is not None:
            vmax = self.config.vmax
        else:
            vmax = data_max + padding

        ax.set_ylim(vmin, vmax)
