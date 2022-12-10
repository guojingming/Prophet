import os
import redis
from cmd_utils import exec_cmd

class RedisInterface:
    def __init__(self, address='127.0.0.1', port='6666'):
        self.server_address = address
        self.server_port = port
        self.redis_client = os.path.join(self.redis_root_path, 'redis-cli.exe')


    def save_value(self, key, value):
        redis.connection()


if __name__ == '__main__':
    redis_interface = RedisInterface()

    result = redis.save_value('test', '123')
    print(result)

    result = redis.load_value('test')
    print(result)

