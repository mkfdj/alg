"""
Visualization module for NCA Trading Bot
Shows training progress, NCA evolution, and performance metrics
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import jax.numpy as jnp

# Try to import plotly, but handle gracefully if not available
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    from IPython.display import display, HTML
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("⚠️  Plotly not available. Interactive visualizations will be disabled.")

# Set style for better looking plots
try:
    plt.style.use('seaborn-v0_8')
except OSError:
    try:
        plt.style.use('seaborn')
    except OSError:
        plt.style.use('default')

try:
    sns.set_palette("husl")
except:
    pass  # Use default colors if seaborn fails


class NCAVisualizer:
    """Visualizer for NCA Trading Bot"""

    def __init__(self):
        self.training_history = {
            'iteration': [],
            'nca_loss': [],
            'ppo_loss': [],
            'portfolio_value': [],
            'sharpe_ratio': [],
            'max_drawdown': [],
            'win_rate': []
        }
        self.nca_evolution_frames = []

    def log_training_step(self, iteration: int, metrics: Dict[str, float]):
        """Log training step metrics"""
        self.training_history['iteration'].append(iteration)
        for key, value in metrics.items():
            if key in self.training_history:
                self.training_history[key].append(value)

    def log_nca_evolution(self, nca_grid: jnp.ndarray):
        """Log NCA grid evolution frame"""
        # Convert to numpy for visualization
        grid_np = np.array(nca_grid)
        if grid_np.ndim == 4:
            grid_np = grid_np[0]  # Take first batch if batched

        self.nca_evolution_frames.append(grid_np.copy())

        # Keep only last 50 frames to save memory
        if len(self.nca_evolution_frames) > 50:
            self.nca_evolution_frames.pop(0)

    def plot_training_progress(self, save_path: Optional[str] = None):
        """Plot comprehensive training progress"""
        if not self.training_history['iteration']:
            return

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('NCA Trading Bot - Training Progress', fontsize=16, fontweight='bold')

        iterations = self.training_history['iteration']

        # NCA Loss
        if 'nca_loss' in self.training_history and self.training_history['nca_loss']:
            axes[0, 0].plot(iterations, self.training_history['nca_loss'], 'b-', linewidth=2)
            axes[0, 0].set_title('NCA Loss', fontweight='bold')
            axes[0, 0].set_xlabel('Iteration')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True, alpha=0.3)

        # PPO Loss
        if 'ppo_loss' in self.training_history and self.training_history['ppo_loss']:
            axes[0, 1].plot(iterations, self.training_history['ppo_loss'], 'r-', linewidth=2)
            axes[0, 1].set_title('PPO Loss', fontweight='bold')
            axes[0, 1].set_xlabel('Iteration')
            axes[0, 1].set_ylabel('Loss')
            axes[0, 1].grid(True, alpha=0.3)

        # Portfolio Value
        if 'portfolio_value' in self.training_history and self.training_history['portfolio_value']:
            axes[0, 2].plot(iterations, self.training_history['portfolio_value'], 'g-', linewidth=2)
            axes[0, 2].set_title('Portfolio Value', fontweight='bold')
            axes[0, 2].set_xlabel('Iteration')
            axes[0, 2].set_ylabel('Value ($)')
            axes[0, 2].grid(True, alpha=0.3)

        # Sharpe Ratio
        if 'sharpe_ratio' in self.training_history and self.training_history['sharpe_ratio']:
            axes[1, 0].plot(iterations, self.training_history['sharpe_ratio'], 'purple', linewidth=2)
            axes[1, 0].set_title('Sharpe Ratio', fontweight='bold')
            axes[1, 0].set_xlabel('Iteration')
            axes[1, 0].set_ylabel('Sharpe Ratio')
            axes[1, 0].grid(True, alpha=0.3)
            axes[1, 0].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Target')
            axes[1, 0].legend()

        # Max Drawdown
        if 'max_drawdown' in self.training_history and self.training_history['max_drawdown']:
            axes[1, 1].plot(iterations, self.training_history['max_drawdown'], 'orange', linewidth=2)
            axes[1, 1].set_title('Maximum Drawdown', fontweight='bold')
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('Drawdown (%)')
            axes[1, 1].grid(True, alpha=0.3)

        # Win Rate
        if 'win_rate' in self.training_history and self.training_history['win_rate']:
            axes[1, 2].plot(iterations, self.training_history['win_rate'], 'teal', linewidth=2)
            axes[1, 2].set_title('Win Rate', fontweight='bold')
            axes[1, 2].set_xlabel('Iteration')
            axes[1, 2].set_ylabel('Win Rate (%)')
            axes[1, 2].grid(True, alpha=0.3)
            axes[1, 2].axhline(y=50, color='red', linestyle='--', alpha=0.7, label='Target')
            axes[1, 2].legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Training progress plot saved to {save_path}")

        plt.show()

    def plot_nca_evolution(self, save_path: Optional[str] = None):
        """Plot NCA grid evolution"""
        if not self.nca_evolution_frames:
            return

        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle('NCA Grid Evolution', fontsize=16, fontweight='bold')

        # Show evolution at different time steps
        frames_to_show = min(4, len(self.nca_evolution_frames))
        frame_indices = np.linspace(0, len(self.nca_evolution_frames)-1, frames_to_show, dtype=int)

        for idx, (ax_idx, frame_idx) in enumerate(zip(axes.flat, frame_indices)):
            grid = self.nca_evolution_frames[frame_idx]

            # Show RGB channels
            if grid.shape[-1] >= 3:
                rgb_grid = grid[:, :, :3]
                # Normalize to [0, 1] for display
                rgb_grid = (rgb_grid - rgb_grid.min()) / (rgb_grid.max() - rgb_grid.min() + 1e-8)

                ax_idx.imshow(rgb_grid)
                ax_idx.set_title(f'NCA Evolution - Step {frame_idx}', fontweight='bold')
                ax_idx.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"🧬 NCA evolution plot saved to {save_path}")

        plt.show()

    def create_animated_nca_evolution(self, save_path: Optional[str] = None):
        """Create animated NCA evolution"""
        if not self.nca_evolution_frames:
            return

        fig, ax = plt.subplots(figsize=(8, 8))
        fig.suptitle('NCA Grid Evolution Animation', fontsize=16, fontweight='bold')

        # Use first frame to determine grid shape
        initial_grid = self.nca_evolution_frames[0]
        if initial_grid.shape[-1] >= 3:
            rgb_grid = initial_grid[:, :, :3]
            rgb_grid = (rgb_grid - rgb_grid.min()) / (rgb_grid.max() - rgb_grid.min() + 1e-8)

        im = ax.imshow(rgb_grid)
        ax.axis('off')
        title = ax.text(0.5, 1.05, '', transform=ax.transAxes, ha='center', fontsize=14, fontweight='bold')

        def animate(frame_idx):
            if frame_idx < len(self.nca_evolution_frames):
                grid = self.nca_evolution_frames[frame_idx]
                if grid.shape[-1] >= 3:
                    rgb_grid = grid[:, :, :3]
                    rgb_grid = (rgb_grid - rgb_grid.min()) / (rgb_grid.max() - rgb_grid.min() + 1e-8)
                    im.set_array(rgb_grid)
                    title.set_text(f'NCA Evolution - Step {frame_idx}')
            return [im, title]

        anim = animation.FuncAnimation(fig, animate, frames=len(self.nca_evolution_frames),
                                       interval=200, blit=True, repeat=True)

        if save_path:
            anim.save(save_path, writer='pillow', fps=5)
            print(f"🎬 NCA animation saved to {save_path}")

        plt.show()

    def plot_portfolio_performance(self, portfolio_values: List[float],
                                   benchmark_values: Optional[List[float]] = None,
                                   save_path: Optional[str] = None):
        """Plot portfolio performance over time"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})

        dates = pd.date_range(start='2020-01-01', periods=len(portfolio_values), freq='D')

        # Portfolio value
        ax1.plot(dates, portfolio_values, label='NCA Trading Bot', linewidth=2, color='blue')

        if benchmark_values:
            ax1.plot(dates, benchmark_values, label='Benchmark', linewidth=2, alpha=0.7, color='red')

        ax1.set_title('Portfolio Performance Over Time', fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Drawdown
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak * 100
        ax2.fill_between(dates, drawdown, 0, alpha=0.3, color='red')
        ax2.plot(dates, drawdown, color='red', linewidth=1)
        ax2.set_title('Drawdown')
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📈 Portfolio performance plot saved to {save_path}")

        plt.show()

    def plot_technical_indicators(self, df: pd.DataFrame, ticker: str, save_path: Optional[str] = None):
        """Plot technical indicators for a stock"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(f'{ticker} - Technical Indicators', fontsize=16, fontweight='bold')

        # Price and Moving Averages
        axes[0].plot(df.index, df['close'], label='Close Price', linewidth=2, color='black')
        if 'sma_20' in df.columns:
            axes[0].plot(df.index, df['sma_20'], label='SMA 20', linewidth=1, color='blue')
        if 'sma_50' in df.columns:
            axes[0].plot(df.index, df['sma_50'], label='SMA 50', linewidth=1, color='red')
        axes[0].set_title('Price and Moving Averages')
        axes[0].set_ylabel('Price ($)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # RSI
        if 'rsi_14' in df.columns:
            axes[1].plot(df.index, df['rsi_14'], label='RSI 14', linewidth=2, color='purple')
            axes[1].axhline(y=70, color='red', linestyle='--', alpha=0.7)
            axes[1].axhline(y=30, color='green', linestyle='--', alpha=0.7)
            axes[1].fill_between(df.index, 70, df['rsi_14'], where=(df['rsi_14'] >= 70),
                                 color='red', alpha=0.3, interpolate=True)
            axes[1].fill_between(df.index, df['rsi_14'], 30, where=(df['rsi_14'] <= 30),
                                 color='green', alpha=0.3, interpolate=True)
        axes[1].set_title('RSI (14)')
        axes[1].set_ylabel('RSI')
        axes[1].set_ylim(0, 100)
        axes[1].grid(True, alpha=0.3)

        # Volume
        if 'volume' in df.columns:
            axes[2].bar(df.index, df['volume'], label='Volume', color='lightblue', alpha=0.7)
        axes[2].set_title('Volume')
        axes[2].set_ylabel('Volume')
        axes[2].set_xlabel('Date')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Technical indicators plot saved to {save_path}")

        plt.show()

    def create_interactive_dashboard(self, save_path: Optional[str] = None):
        """Create interactive dashboard using Plotly"""
        if not PLOTLY_AVAILABLE:
            print("⚠️  Plotly not available. Falling back to matplotlib dashboard...")
            self.plot_training_progress(save_path=save_path)
            return

        if not self.training_history['iteration']:
            return

        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('NCA Loss', 'PPO Loss', 'Portfolio Value',
                          'Sharpe Ratio', 'Max Drawdown', 'Win Rate'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}, {"secondary_y": False}]]
        )

        iterations = self.training_history['iteration']

        # Add traces
        traces = [
            ('nca_loss', 'NCA Loss', 'blue', 0, 0),
            ('ppo_loss', 'PPO Loss', 'red', 0, 1),
            ('portfolio_value', 'Portfolio Value', 'green', 0, 2),
            ('sharpe_ratio', 'Sharpe Ratio', 'purple', 1, 0),
            ('max_drawdown', 'Max Drawdown', 'orange', 1, 1),
            ('win_rate', 'Win Rate', 'teal', 1, 2)
        ]

        for metric_name, title, color, row, col in traces:
            if metric_name in self.training_history and self.training_history[metric_name]:
                fig.add_trace(
                    go.Scatter(x=iterations, y=self.training_history[metric_name],
                             mode='lines', name=title, line=dict(color=color, width=2)),
                    row=row+1, col=col+1
                )

        fig.update_layout(
            title_text="NCA Trading Bot - Interactive Training Dashboard",
            title_x=0.5,
            showlegend=True,
            height=800,
            template="plotly_white"
        )

        if save_path:
            fig.write_html(save_path)
            print(f"📊 Interactive dashboard saved to {save_path}")

        fig.show()

# Global visualizer instance
visualizer = NCAVisualizer()