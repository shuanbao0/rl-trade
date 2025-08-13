"""
Base visualization class

Provides unified base interface and common functionality for all visualization components.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.style as style
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import logging
from pathlib import Path

# Set font style for international display
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']  # Support international character display
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display issue
sns.set_style("whitegrid")
sns.set_palette("husl")


class BaseVisualizer:
    """
    Base visualization class
    
    Provides common functionality for chart creation, style settings, file saving, etc.
    All specific visualization classes should inherit from this class.
    """
    
    def __init__(self, 
                 output_dir: str = "visualizations",
                 style_theme: str = "seaborn-v0_8",
                 figure_size: Tuple[int, int] = (12, 8),
                 dpi: int = 300,
                 save_formats: List[str] = ['png', 'pdf']):
        """
        Initialize base visualizer
        
        Args:
            output_dir: Image output directory
            style_theme: matplotlib style theme
            figure_size: Default figure size
            dpi: Image resolution
            save_formats: List of save formats
        """
        self.output_dir = Path(output_dir)
        self.style_theme = style_theme
        self.figure_size = figure_size
        self.dpi = dpi
        self.save_formats = save_formats
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置日志
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 设置matplotlib样式
        try:
            plt.style.use(self.style_theme)
        except OSError:
            # 如果主题不存在，使用默认样式
            plt.style.use('default')
            self.logger.warning(f"Style theme '{self.style_theme}' not found, using default style")
        
        # Color scheme
        self.colors = {
            'profit': '#2E8B57',      # Sea green - profit
            'loss': '#DC143C',        # Deep red - loss
            'neutral': '#4682B4',     # Steel blue - neutral
            'warning': '#FF8C00',     # Orange - warning
            'info': '#1E90FF',        # Blue - info
            'success': '#32CD32',     # Lime green - success
            'primary': '#6495ED',     # Cornflower blue - primary
            'secondary': '#9370DB',   # Purple - secondary
            'background': '#F5F5F5',  # Smoke white - background
            'grid': '#DCDCDC'         # Light gray - grid
        }
    
    def create_figure(self, 
                      figsize: Optional[Tuple[int, int]] = None,
                      subplots: Tuple[int, int] = (1, 1)) -> Tuple[plt.Figure, plt.Axes]:
        """
        Create matplotlib figure and axes objects
        
        Args:
            figsize: Figure size
            subplots: Subplot layout (rows, cols)
            
        Returns:
            tuple: (figure, axes)
        """
        if figsize is None:
            figsize = self.figure_size
            
        fig, axes = plt.subplots(subplots[0], subplots[1], 
                                figsize=figsize, 
                                dpi=self.dpi,
                                facecolor='white')
        
        # Handle both single and multiple subplot cases uniformly
        if subplots == (1, 1):
            axes = [axes] if not isinstance(axes, (list, np.ndarray)) else axes.flatten()
        elif subplots[0] == 1 or subplots[1] == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
            
        return fig, axes
    
    def format_axis(self, ax: plt.Axes, 
                    title: str = "",
                    xlabel: str = "",
                    ylabel: str = "",
                    grid: bool = True,
                    legend: bool = True) -> None:
        """
        Format chart axes
        
        Args:
            ax: matplotlib axes object
            title: Chart title
            xlabel: X-axis label
            ylabel: Y-axis label
            grid: Whether to show grid
            legend: Whether to show legend
        """
        if title:
            ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=12)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=12)
            
        if grid:
            ax.grid(True, alpha=0.3, color=self.colors['grid'])
            
        if legend:
            ax.legend(frameon=True, fancybox=True, shadow=True)
            
        # Set axis style
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(0.5)
        ax.spines['bottom'].set_linewidth(0.5)
    
    def save_figure(self, fig: plt.Figure, 
                    filename: str,
                    subdir: str = "",
                    tight_layout: bool = True) -> List[str]:
        """
        Save chart to file
        
        Args:
            fig: matplotlib figure object
            filename: Filename (without extension)
            subdir: Subdirectory name
            tight_layout: Whether to use tight layout
            
        Returns:
            List[str]: List of saved file paths
        """
        if tight_layout:
            fig.tight_layout()
            
        # Create subdirectory
        save_dir = self.output_dir
        if subdir:
            save_dir = save_dir / subdir
            save_dir.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        
        # Save in multiple formats
        for fmt in self.save_formats:
            file_path = save_dir / f"{filename}.{fmt}"
            try:
                fig.savefig(file_path, 
                           format=fmt,
                           dpi=self.dpi,
                           bbox_inches='tight',
                           facecolor='white',
                           edgecolor='none')
                saved_files.append(str(file_path))
                self.logger.info(f"Chart saved: {file_path}")
            except Exception as e:
                self.logger.error(f"Failed to save chart ({fmt}): {e}")
        
        return saved_files
    
    def format_currency(self, value: float, 
                       symbol: str = "$",
                       decimals: int = 2) -> str:
        """
        Format currency display
        
        Args:
            value: Numeric value
            symbol: Currency symbol
            decimals: Number of decimal places
            
        Returns:
            str: Formatted currency string
        """
        if abs(value) >= 1e6:
            return f"{symbol}{value/1e6:.1f}M"
        elif abs(value) >= 1e3:
            return f"{symbol}{value/1e3:.1f}K"
        else:
            return f"{symbol}{value:.{decimals}f}"
    
    def format_percentage(self, value: float, 
                         decimals: int = 2,
                         show_sign: bool = True) -> str:
        """
        Format percentage display
        
        Args:
            value: Numeric value (0.1 = 10%)
            decimals: Number of decimal places
            show_sign: Whether to show positive/negative sign
            
        Returns:
            str: Formatted percentage string
        """
        percentage = value * 100
        sign = "+" if show_sign and percentage > 0 else ""
        return f"{sign}{percentage:.{decimals}f}%"
    
    def create_time_series(self, data: List[float], 
                          timestamps: Optional[List] = None,
                          freq: str = '5min') -> pd.Series:
        """
        Create time series data
        
        Args:
            data: Data list
            timestamps: Timestamp list (optional)
            freq: Time frequency
            
        Returns:
            pd.Series: Time series
        """
        if timestamps is None:
            # Create default time index
            start_time = datetime.now() - timedelta(minutes=len(data) * 5)
            timestamps = pd.date_range(start=start_time, 
                                     periods=len(data), 
                                     freq=freq)
        
        return pd.Series(data, index=timestamps)
    
    def add_watermark(self, ax: plt.Axes, 
                     text: str = "TensorTrade",
                     alpha: float = 0.1) -> None:
        """
        Add watermark
        
        Args:
            ax: matplotlib axes object
            text: Watermark text
            alpha: Transparency
        """
        ax.text(0.5, 0.5, text,
                transform=ax.transAxes,
                fontsize=50,
                color='gray',
                alpha=alpha,
                ha='center',
                va='center',
                rotation=30,
                zorder=0)
    
    def get_color_by_value(self, value: float, 
                          threshold: float = 0.0) -> str:
        """
        Get color based on value
        
        Args:
            value: Numeric value
            threshold: Threshold value
            
        Returns:
            str: Color code
        """
        if value > threshold:
            return self.colors['profit']
        elif value < threshold:
            return self.colors['loss']
        else:
            return self.colors['neutral']
    
    def create_summary_table(self, data: Dict[str, Any],
                           title: str = "Summary Statistics") -> plt.Figure:
        """
        Create summary statistics table
        
        Args:
            data: Data dictionary
            title: Table title
            
        Returns:
            plt.Figure: Table figure
        """
        fig, ax = plt.subplots(figsize=(10, len(data) * 0.5 + 2))
        ax.axis('tight')
        ax.axis('off')
        
        # 创建表格数据
        table_data = []
        for key, value in data.items():
            if isinstance(value, float):
                if 'rate' in key.lower() or 'ratio' in key.lower() or '%' in str(value):
                    formatted_value = self.format_percentage(value if value <= 1 else value/100)
                elif '$' in str(value) or 'balance' in key.lower() or 'value' in key.lower():
                    formatted_value = self.format_currency(value)
                else:
                    formatted_value = f"{value:.4f}"
            else:
                formatted_value = str(value)
            
            table_data.append([key, formatted_value])
        
        # 创建表格
        table = ax.table(cellText=table_data,
                        colLabels=['指标', '数值'],
                        cellLoc='left',
                        loc='center',
                        colWidths=[0.6, 0.4])
        
        # 设置表格样式
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 1.8)
        
        # 设置标题
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        
        return fig
    
    def close_all_figures(self):
        """关闭所有打开的图形"""
        plt.close('all')
        
    def __enter__(self):
        """上下文管理器入口"""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close_all_figures()