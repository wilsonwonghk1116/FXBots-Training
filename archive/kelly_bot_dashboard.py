import sys
import json
import time
from pathlib import Path
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QTableWidget, QTableWidgetItem, 
    QHeaderView, QHBoxLayout, QFrame, QScrollArea
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QColor, QPalette

# --- Futuristic Neon Style ---
NEON_COLOR = '#00ffe7'
DARK_BG = '#0a0a0a'
ACCENT_COLOR = '#ff00c8'
SUCCESS_COLOR = '#00ff00'
DANGER_COLOR = '#ff0044'
WARNING_COLOR = '#ffaa00'
FONT_FAMILY = 'Consolas, Courier New, monospace'

class NeonLabel(QLabel):
    def __init__(self, text, size=18, color=NEON_COLOR, bold=True):
        super().__init__(text)
        font = QFont(FONT_FAMILY, size)
        font.setBold(bold)
        self.setFont(font)
        self.setStyleSheet(f"color: {color}; background: transparent; border: none;")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

class NeonFrame(QFrame):
    def __init__(self, border_color=NEON_COLOR):
        super().__init__()
        self.setStyleSheet(f"""
            QFrame {{
                background: {DARK_BG}; 
                border: 2px solid {border_color}; 
                border-radius: 8px;
                margin: 5px;
                padding: 10px;
            }}
        """)

class AnimatedTable(QTableWidget):
    def __init__(self, rows, cols):
        super().__init__(rows, cols)
        self.setStyleSheet(f"""
            QTableWidget {{
                background-color: {DARK_BG};
                color: {NEON_COLOR};
                gridline-color: #333;
                font-family: {FONT_FAMILY};
                font-size: 14px;
                border: 2px solid {NEON_COLOR};
                border-radius: 8px;
            }}
            QTableWidget::item {{
                padding: 8px;
                border-bottom: 1px solid #333;
            }}
            QTableWidget::item:selected {{
                background-color: {ACCENT_COLOR};
                color: white;
            }}
            QHeaderView::section {{
                background-color: {ACCENT_COLOR};
                color: white;
                padding: 10px;
                border: none;
                font-weight: bold;
                font-size: 16px;
            }}
        """)

class BotDashboard(QWidget):
    def __init__(self, results_file: str):
        super().__init__()
        self.results_file = Path(results_file)
        self.setWindowTitle("Kelly Monte Bots - Futuristic Top 20 Dashboard")
        self.setGeometry(100, 100, 1400, 1000)
        self.setStyleSheet(f"background: {DARK_BG};")
        
        # Main layout
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Title
        self.title = NeonLabel("🤖 TOP 20 KELLY MONTE BOTS - REAL TIME PERFORMANCE", 
                              size=24, color=ACCENT_COLOR)
        self.layout.addWidget(self.title)

        # Status and stats row
        stats_frame = NeonFrame(NEON_COLOR)
        stats_layout = QHBoxLayout()
        stats_frame.setLayout(stats_layout)
        
        self.fleet_stats = NeonLabel("Fleet Status: Loading...", size=14, color=SUCCESS_COLOR)
        self.last_update = NeonLabel("Last Update: Never", size=12, color=WARNING_COLOR)
        
        stats_layout.addWidget(self.fleet_stats)
        stats_layout.addWidget(self.last_update)
        self.layout.addWidget(stats_frame)

        # Main table
        self.table = AnimatedTable(20, 4)  # Changed from 10 to 20 rows
        self.table.setHorizontalHeaderLabels([
            "🏆 Rank & Bot", "💰 Total Capital", "📈 Return %", "🔥 Recent Trades (Last 5 PnL)"
        ])
        
        # Set column widths
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed) 
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        
        self.table.setColumnWidth(0, 200)
        self.table.setColumnWidth(1, 150)
        self.table.setColumnWidth(2, 120)
        
        self.layout.addWidget(self.table)

        # Status bar
        self.status = NeonLabel("🔄 Initializing dashboard...", size=14, color=NEON_COLOR)
        self.layout.addWidget(self.status)

        # Timer for live updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_dashboard)
        self.timer.start(2000)  # Update every 2 seconds

        # Load initial data
        self.update_dashboard()

    def format_currency(self, amount):
        """Format currency with color coding"""
        if amount >= 0:
            return f"${amount:,.2f}"
        else:
            return f"-${abs(amount):,.2f}"

    def get_color_for_pnl(self, pnl):
        """Get color based on P&L value"""
        if pnl > 0:
            return SUCCESS_COLOR
        elif pnl < 0:
            return DANGER_COLOR
        else:
            return WARNING_COLOR

    def update_dashboard(self):
        if not self.results_file.exists():
            self.status.setText(f"❌ No results file found: {self.results_file}")
            self.status.setStyleSheet(f"color: {DANGER_COLOR};")
            return

        try:
            with open(self.results_file, 'r') as f:
                data = json.load(f)
        except Exception as e:
            self.status.setText(f"❌ Error loading data: {e}")
            self.status.setStyleSheet(f"color: {DANGER_COLOR};")
            return

        # Get fleet performance data
        fleet_perf = data.get('fleet_performance', {})
        bot_metrics = data.get('bot_metrics', [])
        
        if not bot_metrics:
            self.status.setText("❌ No bot metrics found in data")
            self.status.setStyleSheet(f"color: {DANGER_COLOR};")
            return

        # Sort by current_equity (total capital) and take top 20
        top_bots = sorted(bot_metrics, key=lambda x: x.get('current_equity', 0), reverse=True)[:20]

        # Update fleet stats
        total_trades = fleet_perf.get('total_trades', 0)
        fleet_pnl = fleet_perf.get('total_pnl', 0)
        win_rate = fleet_perf.get('fleet_win_rate', 0) * 100
        
        fleet_text = f"📊 Fleet: {len(bot_metrics)} bots | 🎯 {total_trades:,} trades | 💰 {self.format_currency(fleet_pnl)} | 🏆 {win_rate:.1f}% win rate"
        self.fleet_stats.setText(fleet_text)
        self.fleet_stats.setStyleSheet(f"color: {self.get_color_for_pnl(fleet_pnl)};")

        # Update table
        for row, bot in enumerate(top_bots):
            bot_id = bot.get('bot_id', row)
            bot_name = f"#{row+1} - Bot {bot_id}"
            capital = bot.get('current_equity', 0)
            return_pct = bot.get('total_return_pct', 0)
            
            # Get recent trades PnL
            trade_history = bot.get('trade_history', [])
            if trade_history:
                recent_trades = trade_history[-5:]  # Last 5 trades
                pnl_list = []
                for trade in recent_trades:
                    pnl = trade.get('pnl', 0)
                    pnl_str = f"${pnl:+.0f}"
                    pnl_list.append(pnl_str)
                pnl_display = " | ".join(pnl_list)
            else:
                pnl_display = "No trades yet"

            # Set table items with colors
            rank_item = QTableWidgetItem(bot_name)
            rank_item.setForeground(QColor(NEON_COLOR))
            
            capital_item = QTableWidgetItem(self.format_currency(capital))
            capital_item.setForeground(QColor(self.get_color_for_pnl(capital - 100000)))  # vs initial
            
            return_item = QTableWidgetItem(f"{return_pct:+.2f}%")
            return_item.setForeground(QColor(self.get_color_for_pnl(return_pct)))
            
            trades_item = QTableWidgetItem(pnl_display)
            trades_item.setForeground(QColor(NEON_COLOR))

            self.table.setItem(row, 0, rank_item)
            self.table.setItem(row, 1, capital_item)
            self.table.setItem(row, 2, return_item)
            self.table.setItem(row, 3, trades_item)

        # Clear unused rows
        for row in range(len(top_bots), 20):  # Changed from 10 to 20
            for col in range(4):
                empty_item = QTableWidgetItem("")
                self.table.setItem(row, col, empty_item)

        # Update status
        current_time = time.strftime('%H:%M:%S')
        self.status.setText(f"✅ Dashboard updated successfully")
        self.status.setStyleSheet(f"color: {SUCCESS_COLOR};")
        self.last_update.setText(f"Last Update: {current_time}")
        self.last_update.setStyleSheet(f"color: {WARNING_COLOR};")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Futuristic Kelly Monte Bot Dashboard")
    parser.add_argument('--results', type=str, default='fleet_results.json', help='Path to fleet results JSON file')
    args = parser.parse_args()

    app = QApplication(sys.argv)
    dashboard = BotDashboard(args.results)
    dashboard.show()
    sys.exit(app.exec())
