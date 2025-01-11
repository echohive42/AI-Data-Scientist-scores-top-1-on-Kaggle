# O1 ML Scientist Automation System

An automated machine learning system that leverages O1 and Claude to iteratively develop, improve, and optimize ML solutions.

## Overview

This system automates the entire machine learning workflow by:

1. Generating ML code using O1
2. Fixing errors using Claude
3. Optimizing performance when needed
4. Tracking progress and improvements across iterations
5. Managing solution versions and submissions
6. NOTE: includes datasets for Spceship Titanic Kaggle challenge: https://www.kaggle.com/competitions/spaceship-titanic/overview

🏆 **Proven Performance**: This AI Data Scientist achieved remarkable success on Kaggle's Spaceship Titanic challenge, ranking 29th out of 2,400+ solutions (top 1%)! 🚀 The system autonomously developed, optimized, and fine-tuned its solution to reach this exceptional performance level. 🌟

⚠️ **IMPORTANT SECURITY WARNING**: This system automatically executes AI-generated code. running any auto-generated code carries inherent risks. Use with caution and creator is not responsible for code outputs and code execution!

## 🎥 Watch How It's Built!

**[Watch the complete build process on Patreon](https://www.patreon.com/posts/how-to-build-o1-112197565?utm_medium=clipboard_copy&utm_source=copyLink&utm_campaign=postshare_creator&utm_content=join_link)**
See exactly how this automation system was created step by step, with detailed explanations and insights into the development process.

![image](https://github.com/user-attachments/assets/05b8fe56-5cc8-4917-b8d2-257060a6f44b)




## ❤️ Support & Get 400+ AI Projects

This is one of 400+ fascinating projects in my collection! [Support me on Patreon](https://www.patreon.com/c/echohive42/membership) to get:

- 🎯 Access to 400+ AI projects (and growing daily!)
  - Including advanced projects like [2 Agent Real-time voice template with turn taking](https://www.patreon.com/posts/2-agent-real-you-118330397)
- 📥 Full source code & detailed explanations
- 📚 1000x Cursor Course
- 🎓 Live coding sessions & AMAs
- 💬 1-on-1 consultations (higher tiers)
- 🎁 Exclusive discounts on AI tools & platforms (up to $180 value)

## Key Features

### 🤖 AI Model Integration

- **O1**: Generates and improves ML solutions
- **Claude**: Handles error fixing and code repairs
- Both models maintain code quality and follow best practices

### ⚡ GPU Acceleration

- Automatic GPU detection and utilization
- Graceful fallback to CPU when GPU is unavailable
- Framework-specific GPU optimizations (PyTorch, TensorFlow, XGBoost, LightGBM)

### ⏱️ Performance Management

- Maximum runtime limit (default: 30 minutes)
- Automatic timeout detection
- Performance optimization suggestions when timeout occurs
- Maintains accuracy while improving efficiency

### 🔄 Iterative Improvement

- Tracks performance metrics across iterations
- Uses previous results to guide improvements
- Maintains history of all solutions and progress reports
- Automated versioning of solutions, reports, and submissions

### 📊 Progress Tracking

- Detailed progress reports in JSON format
- Cross-validation scores tracking
- Feature importance analysis
- Model performance metrics
- Execution logs with timestamps

### 🛠️ Error Handling

- Intelligent error vs. warning detection
- Automatic error fixing with Claude
- Missing package installation handling
- Clear error reporting and logging

## File Structure

```
project/
├── o1_ml_scientist.py      # Main automation script
├── solution.py             # Current ML solution
├── progress_report.json    # Current progress metrics
├── submission.csv          # Current submission file
├── execution_outputs.txt   # Execution logs
├── older_solutions/        # Version history
│   ├── solution_1.py
│   ├── progress_report_1.json
│   ├── submission_1.csv
│   └── ...
```

## Configuration

Key configurable parameters:

```python
ITERATIONS = 50                # Maximum iterations
MAX_RUNTIME_MINUTES = 30      # Maximum runtime per solution
CLAUDE_MODEL = "claude-3-5-sonnet-20241022"
O1_MODEL = "o1"
```

## Progress Report Format

```json
{
    "cross_validation_scores": [...],
    "mean_cv_accuracy": float,
    "feature_importance": {
        "feature1": importance1,
        "feature2": importance2,
        ...
    },
    "model_parameters": {...},
    "execution_time": float
}
```

## Error Handling Process

1. **Code Generation**: O1 generates ML solution
2. **Execution**: Code runs with timeout monitoring
3. **Error Detection**: System distinguishes between errors and warnings
4. **Error Fixing**: Claude fixes errors while maintaining core functionality
5. **Performance Optimization**: O1 optimizes slow-running solutions
6. **Verification**: System verifies fixes and optimizations

## Best Practices Enforced

1. GPU utilization when available
2. Proper train/test splitting
3. Cross-validation for model evaluation
4. Feature importance analysis
5. Progress tracking and logging
6. Code efficiency and readability
7. UTF-8 encoding for file operations
8. Proper error handling and reporting

## Limitations

1. Maximum runtime constraint
2. Model-specific GPU support
3. Dependent on API availability
4. Resource intensive for large datasets

## Requirements

- Python 3.x
- OpenAI API access
- Anthropic API access
- Required Python packages:
  - openai
  - anthropic
  - pandas
  - numpy
  - scikit-learn
  - torch (optional for GPU)
  - termcolor
  - other ML frameworks as needed

## Usage

1. Set up API keys as environment variables
2. Prepare your dataset (train.csv and test.csv)
3. Create additional_info.txt with problem description
4. Run the main script:

```bash
python o1_ml_scientist.py
```

## Output

1. **solution.py**: Current ML solution
2. **progress_report.json**: Performance metrics
3. **submission.csv**: Predictions
4. **execution_outputs.txt**: Detailed logs
5. Version history in older_solutions/

## Monitoring

- Real-time execution feedback
- Color-coded status messages
- Detailed error reporting
- Progress tracking across iterations
- Performance metrics logging
