FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ backend/
COPY data/ data/

# data/telco.csv (the real IBM Telco dataset) is already committed and
# copied in above -- this only generates a synthetic fallback if it's
# somehow missing, and never overwrites a real file that's already there.
RUN test -f data/telco.csv || python data/generate_synthetic_data.py

EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
