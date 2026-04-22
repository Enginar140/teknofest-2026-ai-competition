"""
UI Panelleri __init__.py
"""
from .dashboard_panel import DashboardPanel
from .file_manager_panel import FileManagerPanel
from .model_selection_panel import ModelSelectionPanel
from .camera_panel import CameraPanel
from .metrics_panel import MetricsPanel
from .comparison_panel import ComparisonPanel
from .server_panel import ServerPanel
from .settings_panel import SettingsPanel

__all__ = [
    'DashboardPanel',
    'FileManagerPanel',
    'ModelSelectionPanel',
    'CameraPanel',
    'MetricsPanel',
    'ComparisonPanel',
    'ServerPanel',
    'SettingsPanel'
]
