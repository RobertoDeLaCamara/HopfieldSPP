import os
import pytest
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
try:
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.set_visible_devices([], 'GPU')
except Exception:
    pass

@pytest.fixture(autouse=True)
def mock_optimization_iterations(monkeypatch):
    from src.train_model import HopfieldLayer
    from src.train_model_improved import ImprovedHopfieldLayer
    from src.train_model_advanced import AdvancedHopfieldLayer
    from src.train_model_ultra import UltraHopfieldLayer
    import tensorflow as tf

    # Patch iterations for all layers
    original_basic_tune = HopfieldLayer.fine_tune_with_constraints
    def fast_basic_tune(self, source, destination, iterations=10):
        return original_basic_tune(self, source, destination, iterations=iterations)

    original_improved_opt = ImprovedHopfieldLayer.optimize
    def fast_improved_opt(self, source, destination, iterations=10, tolerance=1e-6):
        return original_improved_opt(self, source, destination, iterations=iterations, tolerance=tolerance)

    original_advanced_opt = AdvancedHopfieldLayer.optimize
    def fast_advanced_opt(self, source, destination, iterations=10, tolerance=1e-6, lr_schedule=None):
        return original_advanced_opt(self, source, destination, iterations=iterations, tolerance=tolerance, lr_schedule=lr_schedule)

    original_ultra_opt = UltraHopfieldLayer.optimize
    def fast_ultra_opt(self, source, destination, iterations=10, tolerance=1e-6):
        return original_ultra_opt(self, source, destination, iterations=iterations, tolerance=tolerance)

    # Patch model.fit to avoid long training sessions
    original_fit = tf.keras.Model.fit
    def fast_fit(self, *args, **kwargs):
        if 'epochs' in kwargs and kwargs['epochs'] > 5:
            kwargs['epochs'] = 5
        return original_fit(self, *args, **kwargs)

    monkeypatch.setattr(HopfieldLayer, "fine_tune_with_constraints", fast_basic_tune)
    monkeypatch.setattr(ImprovedHopfieldLayer, "optimize", fast_improved_opt)
    monkeypatch.setattr(AdvancedHopfieldLayer, "optimize", fast_advanced_opt)
    monkeypatch.setattr(UltraHopfieldLayer, "optimize", fast_ultra_opt)
    monkeypatch.setattr(tf.keras.Model, "fit", fast_fit)
