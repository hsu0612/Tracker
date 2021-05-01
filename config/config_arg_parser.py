import configparser
config = configparser.ConfigParser()

# 建立設定區段
config['owner'] = {'name': 'John Doe',
                   'organization': 'Acme Widgets Inc.'}

# 不同的建立設定區段方式
config['database'] = {}
config['database']['server'] = '192.168.2.62'

# 不同的建立設定區段方式
db = config['database']
db['port'] = '143'
db['file'] = 'C:\payroll.dat'

# 寫入 INI 檔案
with open('./example.ini', 'w') as configfile:
  config.write(configfile)