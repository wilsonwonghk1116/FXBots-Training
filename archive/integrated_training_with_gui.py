#!/usr/bin/env python3
"""
Integrated Training System with Real-Time GUI
Combines Ray distributed training with live performance monitoring dashboard
Shows top 20 bots ranked by total capital in real-time
"""

import sys
import os
import time
import json
import threading
import queue
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

# PyQt6 imports for GUI
try:
    from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                QHBoxLayout, QTableWidget, QTableWidgetItem, 
                                QPushButton, QLabel, QTextEdit, QSplitter,
                                QHeaderView, QFrame, QGridLayout, QProgressBar)
    from PyQt6.QtCore import QTimer, QThread, pyqtSignal, Qt, QPropertyAnimation, QRect
    from PyQt6.QtGui import QFont, QPalette, QColor, QPixmap, QPainter, QBrush
    PYQT_AVAILABLE = True
except ImportError:
    PYQT_AVAILABLE = False
    print("PyQt6 not available. Please install with: pip install PyQt6")

# Ray imports
try:
    import ray
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False
    print("Ray not available. Please install with: pip install ray[default]")

@dataclass
class BotPerformance:
    """Bot performance data structure"""
    bot_id: int
    total_capital: float
    total_pnl: float
    win_rate: float
    total_trades: int
    sharpe_ratio: float
    max_drawdown: float
    last_update: datetime

class TrainingMonitor(QThread):
    """Background thread to monitor training progress"""
    
    # Signals for GUI updates
    performance_updated = pyqtSignal(list)  # List of BotPerformance
    training_status_updated = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.is_running = False
        self.training_process = None
        self.results_file = "fleet_results.json"
        
    def start_training(self):
        """Start the Ray training process"""
        try:
            # Start Ray training in background
            cmd = [sys.executable, "ray_kelly_ultimate_75_percent.py"]
            self.training_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            self.is_running = True
            self.training_status_updated.emit("Training started successfully")
            
        except Exception as e:
            self.error_occurred.emit(f"Failed to start training: {str(e)}")
    
    def stop_training(self):
        """Stop the training process"""
        self.is_running = False
        if self.training_process:
            self.training_process.terminate()
            self.training_status_updated.emit("Training stopped")
    
    def run(self):
        """Main monitoring loop"""
        while self.is_running:
            try:
                # Check if results file exists and load performance data
                if os.path.exists(self.results_file):
                    performance_data = self.load_performance_data()
                    if performance_data:
                        self.performance_updated.emit(performance_data)
                
                # Check training process status
                if self.training_process:
                    poll = self.training_process.poll()
                    if poll is not None:
                        if poll == 0:
                            self.training_status_updated.emit("Training completed successfully")
                        else:
                            self.training_status_updated.emit(f"Training failed with code {poll}")
                        self.is_running = False
                
                # Sleep for update interval
                self.msleep(2000)  # Update every 2 seconds
                
            except Exception as e:
                self.error_occurred.emit(f"Monitoring error: {str(e)}")
                self.msleep(5000)  # Wait longer on error
    
    def load_performance_data(self) -> List[BotPerformance]:
        """Load performance data from results file"""
        try:
            with open(self.results_file, 'r') as f:
                data = json.load(f)
            
            bot_performances = []
            
            # Extract bot metrics from fleet data
            if 'bot_metrics' in data:
                for bot_data in data['bot_metrics']:
                    if bot_data and 'current_equity' in bot_data:
                        performance = BotPerformance(
                            bot_id=bot_data.get('bot_id', 0),
                            total_capital=bot_data.get('current_equity', 100000.0),
                            total_pnl=bot_data.get('total_pnl', 0.0),
                            win_rate=bot_data.get('win_rate', 0.0),
                            total_trades=bot_data.get('total_trades', 0),
                            sharpe_ratio=bot_data.get('sharpe_ratio', 0.0),
                            max_drawdown=bot_data.get('max_drawdown', 0.0),
                            last_update=datetime.now()
                        )
                        bot_performances.append(performance)
            
            # Sort by total capital (descending) and return top 20
            bot_performances.sort(key=lambda x: x.total_capital, reverse=True)
            return bot_performances[:20]
            
        except Exception as e:
            print(f"Error loading performance data: {e}")
            return []

class AnimatedTable(QTableWidget):
    """Enhanced table widget with animations and styling"""
    
    def __init__(self, rows, columns):
        super().__init__(rows, columns)
        self.setup_style()
        
    def setup_style(self):
        """Set up consistent blue background with white text for better readability"""
        self.setStyleSheet("""
            QTableWidget {
                background-color: #1a1a2e;
                gridline-color: #00ffff;
                color: #ffffff;
                font-family: 'Courier New', monospace;
                font-size: 12px;
                border: 2px solid #00ffff;
                border-radius: 8px;
            }
            QTableWidget::item {
                background-color: #16213e;
                border: 1px solid #0e3460;
                padding: 10px;
                color: #ffffff;
            }
            QTableWidget::item:selected {
                background-color: #0f4c75;
                color: #ffffff;
            }
            QTableWidget::item:hover {
                background-color: #1f4e79;
                color: #ffffff;
            }
            QHeaderView::section {
                background-color: #0e1929;
                color: #00ffff;
                font-weight: bold;
                font-size: 13px;
                border: 1px solid #00ffff;
                padding: 12px;
            }
        """)
        
        # Set headers
        headers = ["Bot ID", "Total Capital ($)", "Total PnL ($)", "Win Rate (%)", 
                  "Trades", "Sharpe Ratio", "Max DD (%)"]
        self.setHorizontalHeaderLabels(headers)
        
        # Configure table - DISABLE alternating row colors for consistent look
        self.setAlternatingRowColors(False)
        self.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.verticalHeader().setVisible(False)
        
        # Auto-resize columns to fit content
        header = self.horizontalHeader()
        for i in range(len(headers)):
            if i == 1:  # Capital column - needs more space
                header.setSectionResizeMode(i, QHeaderView.ResizeMode.Stretch)
            elif i == 2:  # P&L column - needs more space  
                header.setSectionResizeMode(i, QHeaderView.ResizeMode.Stretch)
            else:
                header.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)
        
        # Set minimum column widths for better readability
        self.setColumnWidth(0, 80)   # Rank
        self.setColumnWidth(3, 100)  # Return %
        self.setColumnWidth(4, 90)   # Win Rate
        self.setColumnWidth(5, 80)   # Trades
        self.setColumnWidth(6, 80)   # Sharpe
        self.setColumnWidth(7, 100)  # Drawdown

class TradingDashboard(QMainWindow):
    """Main dashboard window"""
    
    def __init__(self):
        super().__init__()
        self.monitor_thread = None
        self.setup_ui()
        self.setup_timers()
        
    def setup_ui(self):
        """Set up the user interface"""
        self.setWindowTitle("Kelly Monte Carlo Fleet - Real-Time Performance Monitor")
        self.setGeometry(100, 100, 1600, 1000)
        
        # Set dark futuristic theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #000000;
                color: #00ffff;
            }
            QPushButton {
                background-color: #001a1a;
                color: #00ffff;
                border: 2px solid #00ffff;
                border-radius: 8px;
                font-size: 14px;
                font-weight: bold;
                padding: 10px 20px;
                margin: 5px;
            }
            QPushButton:hover {
                background-color: #003333;
                color: #ffffff;
            }
            QPushButton:pressed {
                background-color: #00ffff;
                color: #000000;
            }
            QLabel {
                color: #00ffff;
                font-family: 'Arial', sans-serif;
                font-size: 12px;
            }
            QTextEdit {
                background-color: #0a0a0a;
                color: #00ffff;
                border: 2px solid #003333;
                border-radius: 8px;
                font-family: 'Courier New', monospace;
                font-size: 10px;
                padding: 10px;
            }
        """)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Title and status
        title_layout = QHBoxLayout()
        
        title_label = QLabel("🚀 KELLY MONTE CARLO FLEET MONITOR 🚀")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #00ffff; margin: 10px;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet("font-size: 14px; color: #ffff00; margin: 10px;")
        
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        title_layout.addWidget(self.status_label)
        
        main_layout.addLayout(title_layout)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("🚀 START TRAINING")
        self.start_button.clicked.connect(self.start_training)
        
        self.stop_button = QPushButton("⏹️ STOP TRAINING")
        self.stop_button.clicked.connect(self.stop_training)
        self.stop_button.setEnabled(False)
        
        self.refresh_button = QPushButton("🔄 REFRESH DATA")
        self.refresh_button.clicked.connect(self.refresh_data)
        
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.refresh_button)
        button_layout.addStretch()
        
        main_layout.addLayout(button_layout)
        
        # Performance metrics summary
        metrics_frame = QFrame()
        metrics_frame.setStyleSheet("""
            QFrame {
                background-color: #001a1a;
                border: 2px solid #00ffff;
                border-radius: 10px;
                margin: 5px;
                padding: 10px;
            }
        """)
        metrics_layout = QGridLayout(metrics_frame)
        
        self.total_bots_label = QLabel("Total Active Bots: 0")
        self.total_capital_label = QLabel("Fleet Total Capital: $0.00")
        self.avg_performance_label = QLabel("Average Performance: 0.00%")
        self.best_bot_label = QLabel("Best Performer: Bot #0 (+0.00%)")
        
        metrics_layout.addWidget(self.total_bots_label, 0, 0)
        metrics_layout.addWidget(self.total_capital_label, 0, 1)
        metrics_layout.addWidget(self.avg_performance_label, 1, 0)
        metrics_layout.addWidget(self.best_bot_label, 1, 1)
        
        main_layout.addWidget(metrics_frame)
        
        # Main content splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Performance table (left side - larger)
        table_widget = QWidget()
        table_layout = QVBoxLayout(table_widget)
        
        table_title = QLabel("🏆 TOP 20 PERFORMERS (by Total Capital) 🏆")
        table_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #00ffff; margin: 10px;")
        table_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        table_layout.addWidget(table_title)
        
        self.performance_table = AnimatedTable(20, 7)
        table_layout.addWidget(self.performance_table)
        
        # Log area (right side - smaller)
        log_widget = QWidget()
        log_layout = QVBoxLayout(log_widget)
        
        log_title = QLabel("📊 TRAINING LOG 📊")
        log_title.setStyleSheet("font-size: 16px; font-weight: bold; color: #00ffff; margin: 10px;")
        log_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        log_layout.addWidget(log_title)
        
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(300)
        log_layout.addWidget(self.log_text)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid #00ffff;
                border-radius: 8px;
                background-color: #001a1a;
                color: #00ffff;
                font-weight: bold;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00ffff, stop:1 #0080ff);
                border-radius: 6px;
            }
        """)
        log_layout.addWidget(self.progress_bar)
        
        # Set splitter proportions (70% table, 30% log)
        splitter.addWidget(table_widget)
        splitter.addWidget(log_widget)
        splitter.setSizes([1120, 480])  # 70% : 30% of 1600px
        
        main_layout.addWidget(splitter)
        
        # Initialize table with empty data
        self.update_table([])
        
        # Add timestamp
        self.last_update_label = QLabel("Last Update: Never")
        self.last_update_label.setStyleSheet("font-size: 10px; color: #666666; margin: 5px;")
        self.last_update_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        main_layout.addWidget(self.last_update_label)
        
    def setup_timers(self):
        """Set up update timers"""
        # Table update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.refresh_data)
        
        # Progress animation timer
        self.progress_timer = QTimer()
        self.progress_timer.timeout.connect(self.animate_progress)
        
    def start_training(self):
        """Start the training process"""
        try:
            self.log_message("🚀 Initializing Kelly Monte Carlo Fleet Training...")
            self.log_message(f"⚡ Target: 75% CPU/GPU utilization across cluster")
            self.log_message(f"🤖 Fleet Size: 2000 bots")
            self.log_message(f"📊 Monte Carlo Scenarios: 300,000 per bot")
            
            # Start monitor thread
            self.monitor_thread = TrainingMonitor()
            self.monitor_thread.performance_updated.connect(self.update_performance)
            self.monitor_thread.training_status_updated.connect(self.update_status)
            self.monitor_thread.error_occurred.connect(self.handle_error)
            
            self.monitor_thread.start_training()
            self.monitor_thread.start()
            
            # Update UI
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            self.status_label.setText("Status: Training Active 🔥")
            
            # Start timers
            self.update_timer.start(3000)  # Update every 3 seconds
            self.progress_timer.start(100)  # Progress animation
            
            self.log_message("✅ Training started successfully!")
            
        except Exception as e:
            self.log_message(f"❌ Failed to start training: {str(e)}")
    
    def stop_training(self):
        """Stop the training process"""
        if self.monitor_thread:
            self.monitor_thread.stop_training()
            self.monitor_thread.wait()
        
        # Update UI
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.status_label.setText("Status: Stopped 🛑")
        
        # Stop timers
        self.update_timer.stop()
        self.progress_timer.stop()
        
        self.log_message("🛑 Training stopped by user")
    
    def refresh_data(self):
        """Manually refresh performance data"""
        if self.monitor_thread and self.monitor_thread.isRunning():
            # Data will be updated automatically by the monitor thread
            pass
        else:
            # Load data manually when not training
            monitor = TrainingMonitor()
            performance_data = monitor.load_performance_data()
            self.update_performance(performance_data)
    
    def update_performance(self, performance_data: List[BotPerformance]):
        """Update the performance display"""
        self.update_table(performance_data)
        self.update_metrics(performance_data)
        self.last_update_label.setText(f"Last Update: {datetime.now().strftime('%H:%M:%S')}")
    
    def update_table(self, performance_data: List[BotPerformance]):
        """Update the performance table with consistent white text on blue background"""
        # Clear and resize table
        self.performance_table.setRowCount(20)
        
        # Fill with performance data
        for i, bot in enumerate(performance_data[:20]):
            # Calculate performance for display (but use consistent white text)
            return_pct = ((bot.total_capital - 100000) / 100000) * 100
            
            # Set cell values with formatting
            items = [
                QTableWidgetItem(f"#{bot.bot_id:04d}"),
                QTableWidgetItem(f"${bot.total_capital:,.2f}"),
                QTableWidgetItem(f"${bot.total_pnl:,.2f}"),
                QTableWidgetItem(f"{bot.win_rate*100:.1f}%"),
                QTableWidgetItem(f"{bot.total_trades:,}"),
                QTableWidgetItem(f"{bot.sharpe_ratio:.3f}"),
                QTableWidgetItem(f"{bot.max_drawdown*100:.1f}%")
            ]
            
            for j, item in enumerate(items):
                # Use consistent white text for better readability
                item.setForeground(QColor("#ffffff"))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                
                # Optional: Add performance indicator with background color
                if j == 2:  # PnL column - add subtle background color hint
                    if return_pct > 10:
                        item.setBackground(QColor("#1a4a1a"))  # Subtle green tint
                    elif return_pct > 0:
                        item.setBackground(QColor("#4a4a1a"))  # Subtle yellow tint
                    elif return_pct < 0:
                        item.setBackground(QColor("#4a1a1a"))  # Subtle red tint
                    else:
                        item.setBackground(QColor("#16213e"))  # Default blue
                else:
                    item.setBackground(QColor("#16213e"))  # Consistent blue background
                
                self.performance_table.setItem(i, j, item)
        
        # Fill remaining rows with empty data
        for i in range(len(performance_data), 20):
            for j in range(7):
                item = QTableWidgetItem("--")
                item.setForeground(QColor("#666666"))  # Muted color for empty rows
                item.setBackground(QColor("#16213e"))   # Consistent blue background
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.performance_table.setItem(i, j, item)
        
        # Refresh column sizing to fit content
        self.performance_table.resizeColumnsToContents()
        
        # Ensure minimum widths are maintained for readability
        self.performance_table.setColumnWidth(0, max(80, self.performance_table.columnWidth(0)))   # Rank
        self.performance_table.setColumnWidth(3, max(100, self.performance_table.columnWidth(3)))  # Return %
        self.performance_table.setColumnWidth(4, max(90, self.performance_table.columnWidth(4)))   # Win Rate
        self.performance_table.setColumnWidth(5, max(80, self.performance_table.columnWidth(5)))   # Trades
        self.performance_table.setColumnWidth(6, max(80, self.performance_table.columnWidth(6)))   # Sharpe
        self.performance_table.setColumnWidth(7, max(100, self.performance_table.columnWidth(7)))  # Drawdown
    
    def update_metrics(self, performance_data: List[BotPerformance]):
        """Update summary metrics"""
        if not performance_data:
            self.total_bots_label.setText("Total Active Bots: 0")
            self.total_capital_label.setText("Fleet Total Capital: $0.00")
            self.avg_performance_label.setText("Average Performance: 0.00%")
            self.best_bot_label.setText("Best Performer: None")
            return
        
        # Calculate metrics
        total_bots = len(performance_data)
        total_capital = sum(bot.total_capital for bot in performance_data)
        avg_return = np.mean([((bot.total_capital - 100000) / 100000) * 100 for bot in performance_data])
        
        best_bot = max(performance_data, key=lambda x: x.total_capital)
        best_return = ((best_bot.total_capital - 100000) / 100000) * 100
        
        # Update labels
        self.total_bots_label.setText(f"Total Active Bots: {total_bots:,}")
        self.total_capital_label.setText(f"Fleet Total Capital: ${total_capital:,.2f}")
        self.avg_performance_label.setText(f"Average Performance: {avg_return:+.2f}%")
        self.best_bot_label.setText(f"Best Performer: Bot #{best_bot.bot_id:04d} ({best_return:+.2f}%)")
    
    def update_status(self, status: str):
        """Update training status"""
        self.status_label.setText(f"Status: {status}")
        self.log_message(f"📢 {status}")
    
    def handle_error(self, error: str):
        """Handle training errors"""
        self.log_message(f"❌ ERROR: {error}")
        self.status_label.setText(f"Status: Error - {error}")
    
    def log_message(self, message: str):
        """Add message to log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        self.log_text.append(formatted_message)
        
        # Auto-scroll to bottom
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def animate_progress(self):
        """Animate progress bar"""
        if self.monitor_thread and self.monitor_thread.is_running:
            # Simulate training progress
            current = self.progress_bar.value()
            if current >= 100:
                self.progress_bar.setValue(0)
            else:
                self.progress_bar.setValue(current + 1)
    
    def closeEvent(self, event):
        """Handle window close"""
        if self.monitor_thread:
            self.stop_training()
        event.accept()

def main():
    """Main application entry point"""
    # Check dependencies
    if not PYQT_AVAILABLE:
        print("❌ PyQt6 is required but not installed.")
        print("Install with: pip install PyQt6")
        return 1
    
    if not RAY_AVAILABLE:
        print("❌ Ray is required but not installed.")
        print("Install with: pip install ray[default]")
        return 1
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("Kelly Monte Carlo Fleet Monitor")
    app.setApplicationVersion("1.0")
    
    # Create and show dashboard
    dashboard = TradingDashboard()
    dashboard.show()
    
    # Log startup message
    dashboard.log_message("🎯 Kelly Monte Carlo Fleet Monitor initialized")
    dashboard.log_message("💡 Ready to start distributed training with real-time monitoring")
    dashboard.log_message("🔥 Click 'START TRAINING' to begin fleet simulation")
    
    # Run application
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
