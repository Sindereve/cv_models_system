FROM redis:7-alpine

COPY redis.conf /usr/local/etc/redis/redis.conf
RUN mkdir -p /data && chown redis:redis /data
EXPOSE 6379

# Запуск Redis с конфигурационным файлом
CMD ["redis-server", "/usr/local/etc/redis/redis.conf"]