import configparser

def test_read():
    config = configparser.ConfigParser()
    config.read('config.ini')
    print(config['DEFAULT']['userlist'].split(','))
    print(config['DEFAULT']['baseline'])
    print(config['DEFAULT']['daynumber'])
    print(config['DEFAULT']['do_gen'])

def write():
    config = configparser.ConfigParser()

    ips = ['42.219.153.7', '42.219.153.89', '42.219.155.56', '42.219.155.26', '42.219.159.194',
            '42.219.152.249', '42.219.159.82', '42.219.159.92', '42.219.159.94', '42.219.158.226']

    config['DEFAULT'] = {'userlist': ','.join(ips),
                        'baseline': 'baseline1',
                        'daynumber': '5',
                        'do_gen': 'no'}

    with open('config.ini', 'w') as configfile:
        config.write(configfile)

if __name__ == "__main__":
    write()
    test_read()