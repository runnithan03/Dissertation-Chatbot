# Dissertation Chatbot

*A chatbot that interprets user queries about equity fund profitability and sustainability using Multiple-Response Regression models.*

---

## **Why I Built It**
- **Problem:** Financial analysis tools often lack user-friendly interfaces for querying complex statistical models.  
- **Motivation:** I wanted to bridge this gap by creating a chatbot that translates people's queries into insights using models from my dissertation.  
- **Goal:** Make complex statistical modelling (Multiple-Response Linear Regression (MRLR), Random Forests, XGBoost) accessible via a conversational interface.

---

## **Architecture Overview**
![Architecture Diagram](images/architecture.png)  

**Key Components:**
- **Chat Interface:** Streamlit (frontend UI)  
- **Backend:** FastAPI (Python) integrated with R models via Plumber or reticulate  
- **Models:** MRLR, Random Forest, XGBoost (trained on equity fund data)  
- **Hosting:** Deployed using Hugging Face Spaces and Gradio 

**Trade-offs:**
- Prioritised **interpretability over raw accuracy** by including MRLR for transparent predictions.
- Chose **FastAPI** for speed and async capabilities instead of Flask.
- Integrated **R models** rather than rewriting all models in Python to preserve the statistical accuracy of dissertation code.

---

## **What I Learned**
- **System Design:** Bridging R and Python environments for model serving requires efficient serialisation and caching to reduce latency.
- **Model Insights:** MRLR provided explainability for feature effects, while XGBoost offered improved predictive accuracy.
- **Soft Skills:** Improved ability to communicate complex modelling ideas to non-technical users through UI design.

---

## **Tech Stack**
| Component  | Technology |
|------------|------------|
| **Frontend**   | Streamlit |
| **Backend**    | Python |
| **Models**     | R (MRLR, Random Forests), Python (XGBoost) |
| **Deployment** | Hugging Face Spaces and Gradio |

---

## **Demo**
- **[Live Demo]([https://huggingface.co/spaces/runnithan03/dissertation-chatbot])** – *Try the chatbot here!*  

---

## **Chatbot in Action** 
![Chatbot Screenshot](images/intermediate.png)  
![Chatbot Outputs](images/model-output.png)  

---

## **How to Run Locally**
Clone the repo and install dependencies:
```bash
git clone https://github.com/runnithan03/equity-fund-chatbot.git
cd equity-fund-chatbot
pip install -r requirements.txt
```

Finally, run the app:
```bash
streamlit run app.py
```
