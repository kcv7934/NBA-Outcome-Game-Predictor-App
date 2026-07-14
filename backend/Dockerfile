FROM python:3.12.7
WORKDIR /backend
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY backend/ .
EXPOSE 5000
CMD ["python", "app.py"]