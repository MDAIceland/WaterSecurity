FROM python:3

COPY water_security .
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app.ipynb .
COPY run.py .

CMD [ "python", "./run.py" ]