FROM ubuntu
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update 
RUN apt-get install -y apache2 
RUN apt-get install -y apache2-utils 
RUN apt-get install -y lsb-release
RUN apt-get install -y net-tools
RUN apt-get clean all
EXPOSE 80
CMD ["apache2ctl", "-D", "FOREGROUND"]
