FROM python:3.10

ENV POLYGON_KEY=PWRjSfmBc9xYnPZMQhA9ukDOuOIId90D
ENV ALPHA_KEY=4RBB1VWACSPNF8DH
ENV TW_KEY=GUukcmQySO7TkElOl1CLYYq6b
ENV TW_SECRET=iwaXSakN4KbMAsjsradSGBg5IOPGomI3RMvbi3kmHv1Kvd1jI9
ENV TW_TOKEN=1514382114564804608-lOQVa5Hj63FBYFcos0D4wQNlm1RUEf
ENV TW_TOKEN_SECRET=3MDTzfYOuYNzCSVzMaWxCLd1ftnOWcFcCQzd2QTr8iYEj
ENV TW_BEARER=AAAAAAAAAAAAAAAAAAAAAITfbQEAAAAAXLq%2BLCIjs62wnHPMvRY0qYvbbqw%3DP4hTjRhOwcC7rO2Fg0xSc1EKzR9cbC8WR7cySe2PZPxUJgTvld
ENV COIN_MKTCAP_KEY=3349b863-7a29-4bc2-8ef3-e5897b394d4d
ENV DS4A_ENV=production


RUN apt-get update && apt-get install -y vim

WORKDIR /usr/src/DS4A

COPY requirements.txt ./
RUN pip install  -r requirements.txt

COPY . .

RUN tar -xzf ta-lib-0.4.0-src.tar.gz

WORKDIR /usr/src/DS4A/ta-lib
RUN ./configure && \
    make && make install
RUN pip install ta-lib

WORKDIR /usr/src/DS4A

EXPOSE 8050

CMD [ "python", "./main.py" ]