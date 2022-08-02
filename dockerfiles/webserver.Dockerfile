FROM ubuntu
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update 
#RUN apt-get install -y apache2 
RUN apt-get install -y apache2-utils 
RUN apt-get install -y lsb-release net-tools git python3 vim python3-pip
RUN apt-get clean all
RUN apt-get install -y git

RUN git clone https://github.com/squeakus/peoplecounter.git
RUN pip3 install websockets
RUN cp /peoplecounter/*.html /var/www/html
RUN cp /peoplecounter/server.py /bin
RUN cp /peoplecounter/run.sh /bin
RUN chmod +x /bin/run.sh

EXPOSE 80
CMD ["/bin/run.sh", "-D", "FOREGROUND"]
