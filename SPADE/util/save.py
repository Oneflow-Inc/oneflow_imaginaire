import os


def saveDict_as_txt(path, things):
    if not os.path.exists('./my_log'):
        os.mkdir('./my_log')

    path = os.path.join('./my_log', path)
    with open(path, 'w') as f:
        for k, v in things.items():
            f.writelines(str(k) + ': ' + str(v)+'\n')
    f.close()
    print('Save ok:'+str(path))

def saveStr_as_txt(path, things):
    path = os.path.join('./my_log', path)
    with open(path, 'a') as f:
        f.writelines(things+'\n')
    f.close()