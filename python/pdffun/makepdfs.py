import os, subprocess, time



svn = "svn co http://ncra.ucd.ie:8080/svn/Jonathan/Documents/pylonGrammars -r"

def run_cmd(cmd,svn=False):
    print cmd
    process = subprocess.Popen(cmd, shell=True,
                               stdout=subprocess.PIPE,
                               stdin=subprocess.PIPE)
    if svn:
        process.wait()
        if process.returncode != 0:
            print "Cannot checkout!\n",cmd
    result = process.communicate()
    
    return result

home =  os.getcwd()
logfile = open('log.txt','r')
revisions = []

for line in logfile:
    revision = line.split(' ')[0]
    revisions.append(revision)
revisions.reverse()

for idx, revision in enumerate(revisions):
    run_cmd('rm -rf pylonGrammars')
    run_cmd(svn + revision, True)
    run_cmd('mkdir pylonGrammars/figures')
    run_cmd('cp figures/*.* pylonGrammars/figures')
    os.chdir(home+'/pylonGrammars')
    run_cmd('rubber -d pylonGrammar.tex')
    newname =  '%03d' % idx
    run_cmd('cp pylonGrammar.pdf ../'+newname+'.pdf')
    os.chdir(home)
