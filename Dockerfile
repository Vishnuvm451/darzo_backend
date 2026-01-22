FROM vrndv/darzo-apii:latest

WORKDIR /app

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
