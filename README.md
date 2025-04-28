# Data Wizards NLP Challenge

## General Overview
This project serves a binary classification machine learning model through a FastAPI application.
It predicts whether a job ad is fraudulent or not, based on its description.

## How to run the project
### 1. Clone the repository
```
git clone https://github.com/lvendrix/data_wizard.git
cd data_wizard
```

### 2. Build the Docker image
```
docker build -t data-wizards-job-ads-api .
```

### 3. Run the Docker container
```
docker run -d -p 8000:8000 data-wizards-job-ads-api
```

### 4. Access the API
Open your browser at http://localhost:8000/docs and test  the **/predict** endpoint from the Swagger UI. 

### 5. Example Request
POST to **/predict**:
```
{
  "description": "Cash In Hand Job (Urgent Staff Required) No Experience Required And Never Any Fees. Work Anytime 1 To 2 Hrs Daily In Free Time. Earn Easily $400 To $500 Extra Per Day. Totally Free To Join & Suitable For All. Take Action & Get Started Today."
}
```
### 6. Example Results
Results can be either
- {"fraudulent":1} for fraudulent job ads
- {"fraudulent":0} for non-fraudulent job ads

Based on the example request given above, we get:
```
{
    "fraudulent": 1
}
```
## Project structure
```
.
├── docs                    # Documentation files (Data Wizards Challenge PDF)
├── models                  # Model files
├── notebooks               # Notebook file (EDA and models training/testing)
├── main.py                 # FastAPI app
├── requirements.txt        # Python dependencies
├── Dockerfile              # Docker instructions
└── README.md               # Project setup
```