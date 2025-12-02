# Deep Learning Project Architecture & Task Distribution

## Project Overview

A comprehensive deep learning project focused on transfer learning for image classification. The project requires fine-tuning pretrained vision models, comparing them with a CNN built from scratch, and explaining model decisions using Grad-CAM.

**Deadline:** 28 December 2025, 23:59

## Architecture Components

### 1. Data Pipeline
- **Exploration & Analysis**: Dataset statistics, class distribution, visualization
- **Preprocessing**: Normalization, resizing, format conversion
- **Augmentation**: Data augmentation techniques (rotation, flip, brightness, etc.)
- **Train/Val/Test Split**: Proper data stratification

### 2. Model Training Pipeline
- **Baseline Pretrained Model**: Default transfer learning setup
- **Hyperparameter Tuning**: 3 configurations for one parameter
- **CNN from Scratch**: Custom architecture for comparison
- **Logging & Checkpointing**: TensorBoard integration, checkpoint saving

### 3. Evaluation Framework
- **Metrics**: Accuracy, precision, recall, F1, confusion matrix
- **Visualization**: Training curves (loss & accuracy), comparison plots
- **Analysis**: Performance comparison across all models

### 4. Explainability Module
- **Grad-CAM Visualizations**: ≥5 examples showing model decisions
- **Error Analysis**: ≥3 misclassified cases with detailed interpretation
- **Feature Analysis**: Discussion of over-reliance and learned features

### 5. Documentation
- **Research Section**: Model selection justification, preprocessing rationale
- **Planning**: Outlined approach with clear objectives
- **Analysis Sections**: 1-2 focused paragraphs per component (avoid generic AI content)

## Technology Stack

- **Deep Learning Framework**: PyTorch (with torchvision for pretrained models)
- **Explainability**: Grad-CAM (or pytorch-grad-cam)
- **Logging**: TensorBoard
- **Visualization**: Matplotlib, Seaborn
- **Computation**: Google Colab (GPU recommended)
- **Data Handling**: Pandas, NumPy

## Project Deliverables

```
submission.zip
├── notebook.ipynb (single comprehensive notebook)
└── saved_models/
    ├── baseline_pretrained/
    │   ├── model_checkpoint.pt
    │   └── logs/
    ├── tuned_variant_[param]_[value]/
    │   ├── model_checkpoint.pt
    │   └── logs/
    └── cnn_scratch/
        ├── model_checkpoint.pt
        └── logs/
```

## Notebook Structure

```
1. Introduction and Plan
   - Research findings on pretrained models
   - Dataset choice with justification
   - Model selection rationale
   - Transfer learning strategy
   - Evaluation metrics planned

2. Data Exploration and Preprocessing
   - Dataset statistics and visualization
   - Class distribution analysis
   - Preprocessing pipeline description
   - Augmentation techniques applied
   - Train/Val/Test split ratios

3. Baseline Model Training
   - Pretrained model architecture selection
   - Transfer learning approach (fine-tuning depth)
   - Training configuration
   - Results and performance analysis

4. Hyperparameter Tuning Experiments
   - Parameter selection justification
   - 3 well-chosen values tested
   - Results comparison for each configuration
   - Selection of best configuration

5. CNN from Scratch
   - Architecture design and justification
   - Training setup (equal epochs to pretrained)
   - Results and performance analysis

6. Grad-CAM and Error Analysis
   - 5+ Grad-CAM visualizations with explanations
   - 3+ misclassified case analysis
   - Feature interpretation and model behavior

7. Final Model Comparison
   - All models on same metrics
   - Training curves for all models
   - Strengths and weaknesses analysis
   - Conclusions drawn from comparisons

8. Conclusions and Future Work
   - Summary of key findings
   - Model comparison conclusions
   - Recommendations for improvements
   - Potential extensions
```

## Key Grading Components

| Component | Weight | Focus Areas |
|-----------|--------|------------|
| Data Preprocessing & Augmentation | 15% | Thorough exploration, correct preprocessing, relevant augmentations |
| Model Training & Optimization | 20% | Baseline, tuning, scratch CNN, logging, reproducibility |
| Explainability & Visual Interpretation | 15% | Grad-CAM visualizations (≥5), error analysis (≥3), critical interpretation |
| Evaluation & Analysis | 25% | Multiple metrics, comparisons, training curves, focused analysis (1-2 paragraphs each) |
| Oral Defense | 25% | Individual understanding of all components (failing this = failing entire project) |

## Critical Success Factors

1. **Balance research with experimentation**: Justify all choices before implementation
2. **Meaningful comparisons**: Train all models for equal epochs; use same train/val split
3. **Quality analysis**: Avoid generic AI-generated content; focus on specific insights
4. **Complete understanding**: Each team member must understand all components
5. **Proper logging**: Use TensorBoard to track every experiment systematically
6. **Clear documentation**: Integrate explanations with code in notebook; no separate reports

---

# Task Distribution: 3 Students

## Student 1: Data & Baseline Model

### Responsibilities
- **Research & Planning** (shared with team)
- **Data Pipeline Development**
- **Baseline Model Implementation & Training**

### Specific Tasks

#### Phase 1: Research & Planning (TEAM)
- [ ] Research 2-3 suitable pretrained models for the chosen dataset
- [ ] Research data preprocessing best practices for CNNs
- [ ] Write research findings in notebook (Introduction and Plan section)
- [ ] Define evaluation metrics appropriate to the task
- [ ] Create overall project plan with timeline

#### Phase 2: Data Exploration & Preprocessing
- [ ] Load and explore dataset
  - [ ] Visualize sample images from each class
  - [ ] Analyze class distribution (identify imbalance if any)
  - [ ] Report dataset statistics (size, resolution, number of classes)
- [ ] Implement preprocessing pipeline
  - [ ] Normalize pixel values appropriately
  - [ ] Resize images to model input size
  - [ ] Handle any format conversion or data cleaning
- [ ] Implement train/validation/test split
  - [ ] Use stratified split if dataset is imbalanced
  - [ ] Document split ratios used
- [ ] Create data loading utilities with proper batching
- [ ] Document all preprocessing steps in notebook

#### Phase 3: Baseline Model Implementation
- [ ] Select and load pretrained model
- [ ] Implement transfer learning strategy
  - [ ] Decide on freezing layers (e.g., freeze backbone, tune final layers)
  - [ ] Replace final classification layer for custom number of classes
  - [ ] Document transfer learning approach
- [ ] Set up baseline training with default parameters
  - [ ] Define optimizer and learning rate
  - [ ] Set batch size and number of epochs (5-15 range)
  - [ ] Set up loss function
- [ ] Implement TensorBoard logging for baseline run
  - [ ] Log training loss and accuracy
  - [ ] Log validation loss and accuracy
  - [ ] Save checkpoints after each epoch
- [ ] Train baseline model
- [ ] Generate baseline results section with:
  - [ ] Final metrics (accuracy, loss on test set)
  - [ ] Training curve visualization
  - [ ] Performance analysis (1-2 paragraphs)

### Deliverables
- Completed "Data Exploration and Preprocessing" section in notebook
- Completed "Baseline Model Training" section in notebook
- saved_models/baseline_pretrained/ folder with checkpoint and TensorBoard logs
- Data loading functions and preprocessing utilities

### Key Files to Create/Modify
- `notebook.ipynb` (sections 1-3)
- `saved_models/baseline_pretrained/model_checkpoint.pt`
- `saved_models/baseline_pretrained/logs/` (TensorBoard event files)

---

## Student 2: Hyperparameter Tuning & Scratch CNN

### Responsibilities
- **Hyperparameter Tuning Experiments**
- **CNN from Scratch Implementation & Training**

### Specific Tasks

#### Phase 1: Hyperparameter Tuning Setup
- [ ] Coordinate with Student 1 to get baseline model and data pipeline
- [ ] Select one hyperparameter to tune
  - [ ] Choose from: learning rate, batch size, dropout rate, weight decay, optimizer type
  - [ ] Justify selection (why this parameter likely affects performance)
  - [ ] Document rationale in notebook
- [ ] Define 3 well-chosen values for the parameter
  - [ ] Use logarithmic spacing for continuous parameters (e.g., learning rate)
  - [ ] Ensure values are meaningfully different (avoid narrow ranges)
  - [ ] Document chosen values and justification

#### Phase 2: Execute Tuning Experiments
- [ ] Run 3 training experiments (one per hyperparameter value)
  - [ ] Keep all other parameters identical to baseline
  - [ ] Train for same number of epochs as baseline (for fair comparison)
  - [ ] Use same train/validation split from Student 1
- [ ] Implement TensorBoard logging for each run
  - [ ] Log training loss and accuracy
  - [ ] Log validation loss and accuracy
  - [ ] Use different run names for comparison
  - [ ] Save checkpoints for each configuration
- [ ] Create comparison visualization
  - [ ] Plot all 3 configurations on same graph
  - [ ] Show convergence behavior and final metrics

#### Phase 3: Analysis & Best Model Selection
- [ ] Compare the 3 configurations
  - [ ] Report metrics for each value
  - [ ] Analyze convergence speed and stability
  - [ ] Identify best-performing configuration
- [ ] Document findings in notebook
  - [ ] 1-2 focused paragraphs explaining results
  - [ ] Discuss effects of parameter changes
  - [ ] Select best configuration with justification

#### Phase 4: CNN from Scratch Implementation
- [ ] Design simple CNN architecture
  - [ ] Similar complexity to examples from course (e.g., 3-5 conv layers)
  - [ ] Include appropriate pooling and fully connected layers
  - [ ] Document architecture rationale
- [ ] Implement CNN model class
  - [ ] Use PyTorch nn.Module
  - [ ] Initialize weights appropriately
- [ ] Set up training pipeline for scratch CNN
  - [ ] Use same optimizer and learning rate as baseline (for fair comparison)
  - [ ] Train for EQUAL number of epochs as pretrained models
  - [ ] Implement TensorBoard logging identical to other models
  - [ ] Save checkpoints
- [ ] Train scratch CNN
- [ ] Generate results section with:
  - [ ] Final metrics on test set
  - [ ] Training curve visualization
  - [ ] Performance analysis (1-2 paragraphs)

### Deliverables
- Completed "Hyperparameter Tuning Experiments" section in notebook
- Completed "CNN from Scratch" section in notebook
- saved_models/tuned_variant_[param]_[value]/ folders (3 variants + best selection)
- saved_models/cnn_scratch/ folder with checkpoint and TensorBoard logs
- Comparison plots and analysis

### Key Files to Create/Modify
- `notebook.ipynb` (sections 4-5)
- `saved_models/tuned_variant_lr_0.001/`, `_0.0001/`, `_0.00001/` (example)
- `saved_models/cnn_scratch/model_checkpoint.pt`
- `saved_models/cnn_scratch/logs/` (TensorBoard event files)

---

## Student 3: Explainability, Evaluation & Final Analysis

### Responsibilities
- **Grad-CAM Implementation & Visualization**
- **Error Analysis**
- **Final Model Comparison & Conclusions**

### Specific Tasks

#### Phase 1: Grad-CAM Setup & Visualization
- [ ] Coordinate with team to get trained baseline model
- [ ] Set up Grad-CAM implementation
  - [ ] Use pytorch-grad-cam or implement custom Grad-CAM
  - [ ] Test on sample images to ensure correct implementation
- [ ] Generate 5+ Grad-CAM visualizations from baseline model
  - [ ] Select diverse examples showing different model behaviors
  - [ ] Include examples where model is correct and confident
  - [ ] Include edge cases and interesting decisions
  - [ ] Visualize images with overlaid heatmaps
- [ ] Write explanations for each visualization
  - [ ] Describe what features the model focuses on
  - [ ] Explain the model's reasoning for predictions
  - [ ] Comment on feature relevance to task

#### Phase 2: Error Analysis & Interpretation
- [ ] Identify at least 3 misclassified samples from baseline model
  - [ ] Can be subset of the 5+ visualizations
  - [ ] Choose diverse failure modes
- [ ] For each misclassified sample:
  - [ ] Show original image and Grad-CAM heatmap
  - [ ] Report predicted class and true class
  - [ ] Generate Grad-CAM visualization
  - [ ] Analyze what features the model focused on
  - [ ] Discuss possible causes of error:
    - [ ] Are the classes visually similar?
    - [ ] Is the model over-relying on certain features?
    - [ ] Is this an ambiguous or mislabeled sample?
  - [ ] Write 2-3 sentence analysis for each case
- [ ] Discuss common patterns in model failures
  - [ ] What types of errors does the model make?
  - [ ] Are there systematic biases?

#### Phase 3: Final Model Comparison
- [ ] Collect results from Students 1 & 2
  - [ ] Baseline pretrained model metrics
  - [ ] Tuned variant metrics (best configuration)
  - [ ] Scratch CNN metrics
- [ ] Create comprehensive comparison visualizations
  - [ ] Side-by-side accuracy/loss comparison
  - [ ] Training curves for all models on same plot
  - [ ] Metrics table (accuracy, precision, recall, F1, etc.)
- [ ] Write comparative analysis
  - [ ] Compare baseline vs. tuned variant: what did tuning achieve?
  - [ ] Compare pretrained vs. scratch CNN: what's the transfer learning benefit?
  - [ ] Discuss trade-offs (speed, complexity, performance)
  - [ ] 1-2 focused paragraphs for each comparison

#### Phase 4: Conclusions & Recommendations
- [ ] Summarize key findings
  - [ ] What was learned about this dataset?
  - [ ] Which model performed best and why?
  - [ ] What role did each component (preprocessing, architecture, tuning) play?
- [ ] Write conclusions section (1-2 paragraphs)
- [ ] Provide recommendations for future work
  - [ ] Alternative architectures to try
  - [ ] Additional hyperparameters to tune
  - [ ] Data augmentation improvements
  - [ ] Robustness testing ideas
  - [ ] Potential extensions (bonus work)

### Deliverables
- Completed "Grad-CAM and Error Analysis" section in notebook
- Completed "Final Model Comparison" section in notebook
- Completed "Conclusions and Future Work" section in notebook
- 5+ Grad-CAM visualization images with explanations
- Error analysis with 3+ misclassified cases
- Comparison plots and comprehensive analysis

### Key Files to Create/Modify
- `notebook.ipynb` (sections 6-8)
- Grad-CAM visualization outputs
- Comparison plots

---

## Team Coordination & Shared Responsibilities

### Joint Activities (All Students)

#### Research & Planning Phase (Week 1-2)
- [ ] Meet to discuss dataset choice and research options
- [ ] Collectively write "Introduction and Plan" section
- [ ] Agree on:
  - [ ] Which dataset to use
  - [ ] Which pretrained models to evaluate
  - [ ] Evaluation metrics
  - [ ] Training parameters (batch size, epochs, etc.)
  - [ ] Data augmentation strategy
- [ ] Divide research tasks for efficiency
- [ ] Document all decisions in notebook

#### Integration & Quality Assurance
- [ ] Regular syncs (at least weekly) to ensure compatibility
- [ ] Share model checkpoints and TensorBoard logs
- [ ] Test that data pipeline works for all models
- [ ] Ensure all sections flow together logically
- [ ] Verify all cells run error-free before submission

#### Oral Defense Preparation (All Students)
- [ ] Each student must understand ALL components:
  - [ ] Data pipeline and preprocessing decisions
  - [ ] All three models (baseline, tuned, scratch)
  - [ ] Hyperparameter tuning methodology
  - [ ] Grad-CAM implementation and results
  - [ ] Evaluation metrics and interpretations
- [ ] Prepare explanations for:
  - [ ] Why specific choices were made
  - [ ] How each component contributes to results
  - [ ] What each visualization shows
  - [ ] How to interpret Grad-CAM results
- [ ] Practice defending methodology and results
- [ ] Be ready to discuss limitations and alternatives

### Communication & Deliverables Handoff

| Phase | Owner | Delivers To | Format |
|-------|-------|------------|--------|
| Research & Planning | All | All | Notebook section + shared doc |
| Data Pipeline | Student 1 | Students 2 & 3 | Data loading functions, TensorBoard setup |
| Baseline Model | Student 1 | Student 3 | Checkpoint, logs, metrics |
| Tuning & Scratch CNN | Student 2 | Student 3 | All checkpoints, logs, metrics |
| Grad-CAM & Analysis | Student 3 | All | Complete notebook sections |
| Final Integration | All | All | Single .ipynb file + saved_models/ |

---

## Timeline Suggestion

**Week 1:** Research, planning, dataset selection, data exploration
**Week 2:** Data preprocessing, baseline model training
**Week 3:** Hyperparameter tuning, scratch CNN training
**Week 4:** Grad-CAM analysis, error analysis, final comparisons
**Week 5:** Integration, quality assurance, oral defense prep
**Week 6:** Final adjustments, submission (by Dec 28, 23:59)

---

## Important Reminders

⚠️ **Oral Defense is Critical**: Failure on oral defense = failing entire project, regardless of notebook score

✓ **Equal Understanding Required**: Each team member must fully grasp all work

✓ **No Separate Report**: Everything goes in the notebook

✓ **Meaningful Analysis**: Avoid generic AI-generated content; focus on specific insights

✓ **Fair Comparisons**: All models trained for same epochs with same data splits

✓ **Reproducibility**: Document all hyperparameters, random seeds, and hardware

✓ **Quality Over Quantity**: Better to have thorough analysis on 3 models than superficial coverage of 5
