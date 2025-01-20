from tkinter import ttk
import tkinter as tk
from datetime import datetime, timedelta
from .config_handler import ConfigHandler

class StatisticsHandler:
    def __init__(self, master: tk.Tk, language: object):
        self.master = master
        self.language = language
        self.config_handler = ConfigHandler()
        self.statistics_window = None

    def show_statistics(self) -> None:
        """Display statistics window"""
        # Check if window exists and is valid
        if self.statistics_window is not None:
            try:
                # Check if window still exists
                self.statistics_window.winfo_exists()
                self.statistics_window.lift()
                return
            except tk.TclError:
                # Window was destroyed, set to None
                self.statistics_window = None
        
        # Create new window if none exists or previous was destroyed
        self.statistics_window = tk.Toplevel(self.master)
        self.statistics_window.title(self.language.STATISTICS_TITLE_TEXT)
        self.statistics_window.geometry("500x400")
        self.statistics_window.resizable(False, False)
        self._create_statistics_ui()

    def _create_statistics_ui(self) -> None:
        """Create statistics window UI"""
        main_frame = ttk.Frame(self.statistics_window, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        stats = self._calculate_statistics()
        
        # Total smiles
        ttk.Label(main_frame, text=self.language.TOTAL_SMILES_TEXT, 
                 font=("Helvetica", 12, "bold")).pack(pady=5)
        ttk.Label(main_frame, text=str(stats['total']), 
                 font=("Helvetica", 12)).pack(pady=5)

        # Time-based statistics
        periods = [
            ('today', self.language.TODAY_SMILES_TEXT),
            ('week', self.language.WEEK_SMILES_TEXT),
            ('month', self.language.MONTH_SMILES_TEXT),
            ('year', self.language.YEAR_SMILES_TEXT)
        ]

        for period, label in periods:
            ttk.Label(main_frame, text=label, 
                     font=("Helvetica", 12, "bold")).pack(pady=5)
            ttk.Label(main_frame, text=str(stats[period]), 
                     font=("Helvetica", 12)).pack(pady=5)

    def _calculate_statistics(self) -> dict:
        """Calculate smile statistics from log file"""
        stats = {
            'total': 0,
            'today': 0,
            'week': 0,
            'month': 0,
            'year': 0
        }

        try:
            with open(self.config_handler.smile_log_path, 'r', encoding='utf-8') as f:
                lines = [line for line in f if '[SMILE DETECTED]' in line]
                
            stats['total'] = len(lines)
            
            now = datetime.now()
            today = now.replace(hour=0, minute=0, second=0, microsecond=0)
            week_ago = today - timedelta(days=7)
            month_ago = today - timedelta(days=30)
            year_ago = today - timedelta(days=365)

            for line in lines:
                try:
                    timestamp_str = line.split('] ')[1].strip()
                    timestamp = datetime.strptime(timestamp_str, '%d.%m.%Y %H:%M:%S')
                    
                    if timestamp >= today:
                        stats['today'] += 1
                    if timestamp >= week_ago:
                        stats['week'] += 1
                    if timestamp >= month_ago:
                        stats['month'] += 1
                    if timestamp >= year_ago:
                        stats['year'] += 1
                except (IndexError, ValueError):
                    continue

        except FileNotFoundError:
            pass

        return stats