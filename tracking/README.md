# MLflow Tracking with Autolog

A simple, modular example demonstrating modern MLflow tracking using **autolog** - the recommended approach for automatic experiment tracking.

## 📁 Project Structure

```
mlflow-misc/
├── tracking/
│   ├── simple_tracking_basic.py    # ✅ Simple autolog example
│   └── README.md                   # This file
├── utils/                          # 🔧 Reusable utility modules
│   ├── mlflow_setup.py            # MLflow configuration & setup
│   ├── data_generation.py         # Synthetic data generation
│   ├── visualization.py           # Plotting & visualization
│   ├── model_evaluation.py        # Model evaluation & metrics
│   ├── loader.py                  # Dynamic module loading
│   └── __init__.py                # Utility imports
└── pyproject.toml                 # UV project configuration
```

## 🚀 Quick Start

### Prerequisites
```bash
# Using uv (recommended)
uv sync

# Or with pip
pip install mlflow scikit-learn pandas numpy matplotlib seaborn
```

### Run the Example
```bash
# Using entry point
uv run mlflow-tracking-example

# Or directly
python tracking/simple_tracking_basic.py
```

## 📊 What the Example Does

The `simple_tracking_basic.py` demonstrates the **complete MLflow workflow** in 6 simple steps:

### **1. 🔧 Setup MLflow**
- Configures tracking URI and experiment
- Enables sklearn autolog for automatic logging

### **2. 📊 Generate Data**
- Creates synthetic regression dataset (500 samples, 5 features)
- Uses utility functions for consistent data generation

### **3. 🏃 Run Experiments**
- Trains 3 RandomForest models with different hyperparameters
- **Autolog automatically captures**: parameters, metrics, model artifacts

### **4. 🏆 Find Best Model**
- Compares RMSE across experiments
- Identifies best performing model

### **5. 📦 Load Model**
- Loads the best model using MLflow model URI
- Demonstrates model persistence and retrieval

### **6. 🔮 Make Predictions**
- Uses loaded model for inference on sample data
- Shows end-to-end workflow completion

## 🤖 MLflow Autolog Benefits

### **Automatic Logging (Zero Code Required)**
```python
# Just enable autolog and train - everything is logged automatically!
mlflow.sklearn.autolog()
model = RandomForestRegressor(n_estimators=100, max_depth=10)
model.fit(X_train, y_train)  # ✨ Parameters, metrics, model all auto-logged!
```

### **What Gets Automatically Logged**
- ✅ **Model parameters** (n_estimators, max_depth, etc.)
- ✅ **Model artifacts** with signatures and input examples
- ✅ **Training metrics** (sklearn provides)
- ✅ **Feature importance** (for tree-based models)
- ✅ **Model metadata** and environment info

### **90% Less Code vs Manual Logging**
```python
# ❌ Manual logging (15+ lines)
mlflow.log_params({'n_estimators': 100, 'max_depth': 10})
mlflow.log_metrics({'rmse': 25.4, 'r2': 0.92})
mlflow.sklearn.log_model(model, "model", signature=sig, input_example=X[:5])
# ... many more lines

# ✅ Autolog (3 lines!)
mlflow.sklearn.autolog()
model = RandomForestRegressor(n_estimators=100, max_depth=10)
model.fit(X_train, y_train)  # Everything auto-logged!
```

## 🔍 Example Output

```
========================================
Simple MLflow Tracking
========================================
✅ MLflow sklearn autolog enabled
🎯 Active experiment: simple-house-prediction
🔢 Generating regression dataset: 500 samples, 5 features

Running 3 experiments...
Experiment 1: RMSE=37.528, R²=0.881
Experiment 2: RMSE=29.407, R²=0.927
Experiment 3: RMSE=28.237, R²=0.933

🏆 Best model: Run 816c1cd4 with RMSE=28.237

📦 Loading best model...
✅ Model loaded from: runs:/816c1cd4.../model

🔮 Making predictions...
Sample predictions:
  Sample 1: Predicted=-50.66, Actual=-38.46
  Sample 2: Predicted=29.22, Actual=6.32
  Sample 3: Predicted=-23.35, Actual=6.62

✅ Complete! View results: mlflow ui
========================================
```

## 🛠️ Utility Modules

The example uses modular utility functions for reusability:

### **`utils/mlflow_setup.py`**
```python
from utils.loader import load_mlflow_setup
mlflow_setup = load_mlflow_setup()

# Setup MLflow with autolog
mlflow_setup.setup_mlflow_tracking(
    experiment_name="my-experiment",
    enable_autolog=True
)
```

### **`utils/data_generation.py`**
```python
from utils.loader import load_data_generation
data_gen = load_data_generation()

# Generate synthetic datasets
df, features = data_gen.generate_regression_data(
    n_samples=1000, n_features=10
)
```

### **`utils/visualization.py`**
```python
from utils.loader import load_visualization
viz = load_visualization()

# Create comprehensive plots
plot_path = viz.create_regression_plots(
    y_true, y_pred, feature_importance, feature_names
)
```

### **`utils/model_evaluation.py`**
```python
from utils.loader import load_model_evaluation
eval_utils = load_model_evaluation()

# Comprehensive model evaluation
metrics = eval_utils.evaluate_regression_model(
    model, X_train, X_test, y_train, y_test
)
```

### **Dynamic Loading with `importlib.util`**
The utilities use dynamic imports for flexibility:
```python
from utils.loader import UtilityLoader

loader = UtilityLoader()
mlflow_setup = loader.load_module("mlflow_setup")
data_gen = loader.load_module("data_generation")
```

## 🌐 MLflow UI

After running the example, explore results in the MLflow UI:

```bash
# Start MLflow UI
mlflow ui

# Visit in browser
open http://localhost:5000
```

### In the MLflow UI you can:
- **Compare experiments** side by side with auto-logged parameters
- **View all metrics** captured by autolog
- **Download model artifacts** with signatures and examples
- **Inspect feature importance** for tree-based models
- **Load models** directly from the UI
- **Track experiment lineage** and reproducibility

## 🔧 Customization

### **Use Your Own Data**
```python
# Replace data generation with your dataset
df = pd.read_csv('your_data.csv')
feature_names = ['feature1', 'feature2', 'feature3']
```

### **Different ML Algorithms**
Autolog works with many frameworks:
```python
# Sklearn
mlflow.sklearn.autolog()
model = RandomForestRegressor()

# XGBoost
mlflow.xgboost.autolog()
model = xgb.XGBRegressor()

# TensorFlow
mlflow.tensorflow.autolog()
model = tf.keras.Sequential([...])
```

### **Custom Hyperparameters**
```python
experiments = [
    {'n_estimators': 50, 'max_depth': 5},
    {'n_estimators': 100, 'max_depth': 10},
    {'n_estimators': 200, 'max_depth': 15},
    # Add your configurations
]
```

## 🎯 Key Features

### **✨ Simplicity**
- **60 lines** of clean, focused code
- **6 clear steps** from setup to inference
- **No complex configurations** required

### **🔧 Modularity**
- **Reusable utilities** for different experiments
- **Dynamic imports** using `importlib.util`
- **Separation of concerns** (setup, data, visualization, evaluation)

### **🚀 Modern MLflow**
- **Pure autolog** workflow (no manual logging)
- **Automatic model signatures** and input examples
- **Production-ready** model artifacts

### **📈 Comprehensive**
- **End-to-end workflow** from training to inference
- **Best model selection** based on metrics
- **Model persistence** and loading demonstration

## 🐛 Troubleshooting

### **MLflow UI Issues**
```bash
# Port already in use
mlflow ui --port 5001

# Permission errors
chmod -R 755 mlruns/
```

### **Environment Issues**
```bash
# Activate environment
source .venv/bin/activate

# Or use uv
uv run python tracking/simple_tracking_basic.py
```

### **Import Errors**
```bash
# Ensure dependencies are installed
uv sync
# or
pip install mlflow scikit-learn pandas numpy matplotlib
```

## 📚 Next Steps

1. **Run the example** and explore the MLflow UI
2. **Try different algorithms** (XGBoost, LightGBM, etc.)
3. **Use your own datasets** and problems
4. **Extend utilities** for your specific needs
5. **Set up remote tracking** for team collaboration
6. **Explore MLflow Model Registry** for production deployment

## 🔗 Resources

- [MLflow Autolog Documentation](https://mlflow.org/docs/latest/tracking.html#automatic-logging)
- [MLflow Tracking Guide](https://mlflow.org/docs/latest/tracking.html)
- [Supported ML Libraries](https://mlflow.org/docs/latest/tracking.html#automatic-logging)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)

---

**Happy experimenting with MLflow!** 🚀✨

> **Pro Tip**: The beauty of autolog is its simplicity - just enable it and train your models. MLflow handles all the tracking automatically, letting you focus on the ML rather than the logging! 🎯