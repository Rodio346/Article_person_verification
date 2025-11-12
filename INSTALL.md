# Installation Guide

This guide will help you set up the Article Person Verification System on your machine.

## Prerequisites

- **Python 3.8 or higher**
- **pip** (Python package installer)
- **Git** (optional, for cloning)

## Installation Methods

### Method 1: Using pip (Recommended)

1. **Clone or download the repository**:
   ```bash
   git clone <repository-url>
   cd Article_person_verification
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**:
   Create a `.env` file in the project root:
   ```bash
   # Windows
   copy .env.example .env

   # macOS/Linux
   cp .env.example .env
   ```

   Edit `.env` and add your API keys:
   ```
   GOOGLE_API_KEY=your_gemini_api_key_here
   MLFLOW_TRACKING_URI=file:./mlruns
   ```

5. **Verify installation**:
   ```bash
   python main.py --help
   ```

### Method 2: Using setup.py (For Development)

1. **Clone the repository and navigate to it**:
   ```bash
   git clone <repository-url>
   cd Article_person_verification
   ```

2. **Install in development mode**:
   ```bash
   pip install -e .
   ```

   This installs the package in editable mode, so changes to the code are immediately reflected.

3. **Set up environment variables** (same as Method 1, step 4)

## Getting API Keys

### Google Gemini API Key

1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Click "Create API Key"
3. Copy the key and add it to your `.env` file

## Verifying Installation

Run a test verification:

```bash
python main.py --name "Test User" --dob "01/01/1990" --text "This is a test article about Test User."
```

You should see the verification process run and results displayed.

## Troubleshooting

### Issue: "ModuleNotFoundError"

**Solution**: Ensure you've activated your virtual environment and installed all requirements:
```bash
pip install -r requirements.txt
```

### Issue: "GOOGLE_API_KEY not found"

**Solution**: Make sure your `.env` file exists and contains:
```
GOOGLE_API_KEY=your_actual_key_here
```

### Issue: Rate limiting errors

**Solution**: Adjust the `BATCH_PROCESSING_DELAY` in `src/config/settings.py`:
```python
BATCH_PROCESSING_DELAY = 5.0  # Increase from 3.0 to 5.0 seconds
```

### Issue: Graph visualization not saving

**Solution**: Install optional dependencies:
```bash
# For PNG visualization (requires system Graphviz)
pip install pygraphviz pydot

# Install Graphviz system package:
# Windows: Download from https://graphviz.org/download/
# macOS: brew install graphviz
# Linux: sudo apt-get install graphviz
```

## Updating

To update to the latest version:

```bash
git pull origin main
pip install -r requirements.txt --upgrade
```

## Uninstalling

To completely remove the system:

1. Deactivate virtual environment:
   ```bash
   deactivate
   ```

2. Delete the project directory:
   ```bash
   # Be careful with this command!
   rm -rf Article_person_verification
   ```

## Next Steps

After installation, check out:
- [README.md](README.md) for usage examples
- [logs/](logs/) for execution logs
- [evaluation_results/](evaluation_results/) for accuracy reports

## Support

For issues or questions:
1. Check the logs in `logs/` directory
2. Review MLflow runs: `mlflow ui`
3. See the README for usage examples
