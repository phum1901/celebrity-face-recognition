FROM python:3.10.4-slim

COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip==22.0.4
RUN pip install -r requirements.txt

COPY recognition ./recognition 
# COPY artifacts ./artifacts
COPY app ./app
COPY rawdata/celebrity-face-recognition/accounts.json ./rawdata/celebrity-face-recognition/accounts.json

EXPOSE 80

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
