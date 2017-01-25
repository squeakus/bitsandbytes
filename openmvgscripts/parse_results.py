import os
import subprocess


def main():
    #copy_meshes()
    copy_clouds()


def copy_meshes():
    for (dir, _, files) in os.walk("./"):
        for f in files:
            if f.endswith('OUTPUT_MESH_CLEAN.ply'):
                path = os.path.join(dir, f)
                print path
                newname = clean_name(path)
                run_cmd('cp '+path+' '+newname)


def copy_clouds():
    for (dir, _, files) in os.walk("./"):
        for f in files:
            if f.endswith('colorized.ply'):
                path = os.path.join(dir, f)
                print path
                newname = clean_name(path)
                run_cmd('cp '+path+' '+newname)


def run_cmd(cmd):
    print cmd
    process = subprocess.Popen(cmd, shell=True,
                               stdout=subprocess.PIPE,
                               stdin=subprocess.PIPE)
    result = process.communicate()
    return result


def clean_name(query):
    query =query.replace('./','')
    query =query.replace('OUTPUT_MESH_CLEAN','')
    query =query.replace('robust','')

    query =query.replace('/',' ')
    query =query.replace('_',' ')

    stopwords = ['result','opencv','mve']
    querywords = query.split()

    resultwords  = [word for word in querywords if word.lower() not in stopwords]
    result = ''.join(resultwords)
    return result


if __name__=='__main__':#
    main()
