
import subprocess

def exec_subproc(cmd, show_info=1):
    if show_info:
        print(' '.join(cmd))
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    if show_info:
        if p.returncode == 0:
            print('[OK] ')
            print (out)
        else:
            print(' [ERROR] ' + ' '.join(cmd))
            print (out)
            print(err)
    return p.returncode


def batch_execute_proc(batch_name):
    f = open(batch_name, 'r')

    for line in f.read().split():
        cmd = line.split(' ')
        exec_subproc(cmd, 1)
    f.close()
