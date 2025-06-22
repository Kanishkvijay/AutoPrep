# AutoPrep – A Multi-Agent AI System for Automated & Guided Data Preprocessing

AutoPrep is a modular, explainable, and user-guided data preprocessing platform built using a multi-agent architecture with LangGraph, Streamlit, and LLMs. It automates tasks like data inspection, cleaning, encoding, scaling, and readiness scoring — all while allowing natural language or rule-based user instructions.

---

## 🚩 Problem Statement

Data preprocessing is one of the most time-consuming and error-prone phases of the ML pipeline. There is a strong need for a system that can:

- Automate standard preprocessing tasks  
- Enable human-in-the-loop guidance  
- Provide explainability for each step  
- Measure preprocessing quality via a readiness score

---

## 🎯 Project Objective

To develop a fully modular AI system that:

- Accepts raw `.csv` input  
- Performs end-to-end preprocessing  
- Incorporates user-defined rules  
- Provides real-time explanations and quality metrics

---

## 🧠 Core Features & Agents

### 1. **CSV Upload & Initial Inspection**
- Upload data via Streamlit UI  
- `Inspection Agent`: Detects types, missing values, imbalance, duplicates, and more  
- Suggests preprocessing plan  

### 2. **Automated Agents**
- `Cleaning Agent`: Handles duplicates, missing values (drop, mode, mean, median)  
- `Encoding Agent`: Label/OneHot encoding for categorical features  
- `Scaling Agent`: Applies Standard or MinMax scaling for numerical columns  

### 3. **User Instruction Agent**
- Accepts natural language (e.g., “Drop column ID”) or JSON-based rule input  
- Converts instructions using LLM or rule interpreter  
- Executes safely in a sandboxed environment  

### 4. **Explainability Agent**
- Explains each step in plain English  
- Example: *“Encoded ‘Gender’ using Label Encoding due to binary category.”*  

### 5. **Model Feedback Agent (Optional)**
- Trains a basic model (logistic regression/tree)  
- Evaluates impact of preprocessing on performance  
- Suggests reprocessing if results are poor  

### 6. **Readiness Score Agent**
- Generates a Data Readiness Score (0–100)  
- Considers missing value resolution, feature balance, encoding quality, etc.  

### 7. **Export & Summary Report**
- Final dataset download  
- Summary includes: transformations, rules, readiness score, and final schema  

---

## ⚙️ Tech Stack

| Component       | Tools/Frameworks                           |
|----------------|---------------------------------------------|
| Frontend       | Streamlit                                   |
| Backend        | Python (Pandas, Scikit-learn, Regex)        |
| LLM Support    | Groq API                                    |
| Agents & Logic | Modular Python classes with LangGraph       |
| Optional Tools | FAISS, LangChain (future enhancement)       |

---

## 🌟 What Makes It Unique

- **Modular multi-agent architecture**  
- **Human-guided preprocessing via natural language**  
- **Transparency-first design (explanations + logs)**  
- **Actionable data readiness scoring**  
- Built for **real-world messy datasets**

---

## 📦 AutoPrep Workflow (Step-by-Step)

1. **CSV Upload** via UI  
2. **Inspection Agent** performs profiling & suggests a plan  
3. **User Instructions** parsed & applied  
4. **Cleaning Agent** handles missing data & duplicates  
5. **Encoding Agent** applies suitable encodings  
6. **Scaling Agent** normalizes numeric features  
7. **Explainability Agent** narrates each transformation  
8. **Model Feedback Agent** (optional) tests output quality  
9. **Readiness Score Agent** scores the dataset  
10. **Export & Summary Report** with steps and stats  

---

## 📂 Output

- Fully cleaned, encoded, and scaled dataset  
- Clear traceability of each action  
- Summary report for audit and reproducibility  
- Preprocessed file ready for modeling or training  
