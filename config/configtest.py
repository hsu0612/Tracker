# 引入 configparser 模組
import configparser

# 建立 ConfigParser
config = configparser.ConfigParser()

# 讀取 INI 設定檔
config.read('./config/example.ini')
# 取得設定值
print(int(config['video']['start']))