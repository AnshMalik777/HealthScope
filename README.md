# HealthScope – Healthcare Data Analytics & Visualisation

HealthScope is a healthcare analytics project designed to explore healthcare data, identify disease patterns, analyse patient demographics, and understand treatment costs through interactive Power BI dashboards and AI-powered question answering.

## Key Features

- **Demographic Analysis:** Explore patient distribution by age, gender, state and region.
- **Time Trend Analysis:** Analyse healthcare trends over time.
- **Disease Distribution:** Examine disease frequency and geographical distribution.
- **Risk Analysis:** Explore health risk factors and patient outcomes.
- **Cost Analysis:** Analyse treatment expenditure and hospital costs.
- **Hospital Insights:** Explore hospital types and healthcare service patterns.
- **AI-powered Assistant:** Ask questions about healthcare data using Gemini AI.

## Technologies Used

- Python
- Pandas
- Flask
- Google Gemini API
- Power BI
- HTML, CSS and JavaScript

## Project Structure

```text
HealthScope/
├── api.py
├── app.py
├── analysis_engine.py
├── gemini_engine.py
├── question_engine.py
├── columns.txt
├── requirements.txt
├── Website_Frontend.html
└── .gitignore
```

## Running Locally

1. Clone this repository.
2. Create and activate a Python virtual environment.
3. Install dependencies using `pip install -r requirements.txt`.
4. Configure your Gemini API key in a local `.env` file.
5. Place your authorised dataset in the required local data folder.
6. Start the Flask backend using `python api.py`.
7. Open `Website_Frontend.html` in your browser.

## Note

The dataset and API credentials are not included in this public repository. Users must configure their own authorised dataset and Gemini API key.

## Project Purpose

Developed as an academic Big Data Healthcare Analysis project to demonstrate data analytics, interactive visualisation and AI-assisted exploration of healthcare information.
