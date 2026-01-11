# 🛒 Hybrid Ecommerce Recommendation System

An end-to-end machine learning solution designed to provide personalized product suggestions. This project handles the "Cold Start" problem by combining user behavioral data with product metadata, deployed via a high-performance FastAPI microservice.



## 🚀 Features
- **Hybrid Engine:** Combines Matrix Factorization (SVD) and Cosine Similarity.
- **Cold Start Solution:** Uses popularity-based fallbacks for new users.
- **Real-time API:** Served via FastAPI with automatic Swagger documentation.
- **Business Dashboard:** Interactive Power BI report for stakeholder insights.

## 🛠️ Tech Stack
- **Python:** Pandas, Scikit-Learn, NumPy, SVD
- **API:** FastAPI, Uvicorn
- **BI:** Power BI (Data Modeling & DAX)
- **Deployment:** Pickle (Model Serialization)

---

## 📂 Project Structure
- `app.py`: The FastAPI deployment script.
- `ecommerce_model.pkl`: The serialized model artifacts.
- `ecommerce_recommendation_dataset.csv`: The cleaned dataset.
- `Report.pbix`: The Power BI dashboard file.
- `requirements.txt`: List of necessary Python libraries.

---

## 📈 Dashboard Insights
The Power BI dashboard provides three key views:
1. **Executive Overview:** High-level KPIs and category performance.
2. **User Demographics:** Segmenting customers by age, income, and membership.
3. **Model Validation:** Live testing of AI outputs against user history.



---

## 💻 How to Run the API
1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt


2. Start the server :
   ```bash
   python -m uvicorn app:app --reload

3. Test the Endpoint:
   Navigate to
   http://127.0.01.:8000/recommend/78517 in your browser.

## 🧠 Technical Methodology
**Data Sparsity:** Addressed 99.97% empty matrix using SVD latent factors.
**Scaling:** Normalization of Income and Age for model training.
**Serialization:** Used Pickle to bundle the SVD matrices and similarity maps for lightweight deployment.

Author: [Mohamed Nawfal M]
