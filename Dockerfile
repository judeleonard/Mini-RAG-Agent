FROM python:3.10-slim

RUN apt-get update

ENV INSTALL_PATH /minimal-rag_app

ENV PYTHONPATH="${INSTALL_PATH}:${PYTHONPATH}"

RUN mkdir -p $INSTALL_PATH

WORKDIR $INSTALL_PATH

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

RUN chmod +x entrypoint.sh

ENTRYPOINT [ "./entrypoint.sh" ]