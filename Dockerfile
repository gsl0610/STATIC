FROM nginx:alpine

# 复制静态文件到 Nginx 默认目录
COPY architecture.html /usr/share/nginx/html/architecture.html
COPY training_monitor.html /usr/share/nginx/html/training_monitor.html
COPY training_monitor.html /usr/share/nginx/html/index.html

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
