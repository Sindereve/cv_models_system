FROM continuumio/miniconda3:latest

WORKDIR /app

COPY environment.yml .
RUN conda env create -f environment.yml && \
    conda clean -afy
ENV PATH /opt/conda/envs/cv-trainer/bin:$PATH

COPY src/ ./src/

ENV PYTHONPATH=/app/src:$PYTHONPATH

EXPOSE 311

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "311"]