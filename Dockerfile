FROM python:3.5

WORKDIR /srv

COPY requirements.txt .

RUN pip install -r requirements.txt && python -m spacy download en_core_web_lg \
    && pip install --upgrade pip && pip install wmd==1.2.10

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
ENV PYTHONPATH "${PYTHONPATH}:/srv" 

CMD ["bash"]
