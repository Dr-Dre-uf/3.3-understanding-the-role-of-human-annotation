Genomic AI Reproducibility Analysis Tool
Overview
This application is designed to analyze and demonstrate the impact of human annotation reproducibility on AI models within the context of basic science, specifically genomic mutation analysis. In fields like bioinformatics, AI systems often require large-scale labeled datasets to predict the effects of genetic mutations on protein structure and function.

However, inconsistencies among researchers—due to experimental conditions or biases—can introduce errors into the training process. This tool provides a framework to quantify those inconsistencies and observe how they affect AI predictive accuracy.

Key Features
Mutation Study Simulation: Generate synthetic datasets where multiple "researchers" classify mutations with adjustable levels of noise and variability.

Intraclass Correlation Coefficient (ICC) Calculation: Measures the level of agreement between different annotators. High ICC values indicate a more reproducible and scientifically rigorous dataset.

AI Performance Impact: Evaluates how the number of annotators and their level of agreement directly influence the Mean Squared Error (MSE) of an AI model.

Consensus-Based Training: Implements a "Consensus Label" (the mean of multiple researchers) to train a Random Forest Regressor, demonstrating how collective human intelligence improves AI reliability.

Scientific Significance
Assessing Consistency: By using the ICC, researchers can determine if their labeling process is stable enough for AI training.

Optimizing Annotator Count: The tool visualizes how increasing the number of researchers can reduce error, helping labs determine the optimal workforce needed for a study.

Ensuring Rigor: This framework helps ensure that AI systems used in genomics are trained on reliable, reproducible data, leading to more trustworthy biological discoveries.

Installation & Setup
1. Prerequisites
Ensure you have Python installed, then install the necessary dependencies:

Bash
pip install streamlit numpy pandas scikit-learn
2. Requirements File
If deploying to Streamlit Cloud, ensure you have a requirements.txt file in your root directory containing:

streamlit

numpy

pandas

scikit-learn

3. Run the App
Navigate to the project directory and run:

Bash
streamlit run streamlit_app.py
Data Privacy Warning
🔒 Important: Do not upload files containing Personally Identifiable Information (PII), sensitive patient records, or private genomic data. This application is intended for educational and algorithmic analysis purposes only.
