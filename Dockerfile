FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .

# Exponha a porta, se necessário (para uso futuro em servidores ou APIs)
# EXPOSE 8080 

CMD ["python", "knowledge-graph.py"]
