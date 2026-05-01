"""
test local commands
"""
#===============================================================================
import os, sys
from click import echo
#===============================================================================
# Make sure that the current directory is the project directory.
# 'make test" and 'pytest' are generally run from the project directory.
# However, if we run/debug this file in eclipse, we end up in lrcmd/tests
cwd = os.getcwd()
if cwd.endswith('tests'):
    echo(f"Changing current working directory"
         f"\n  from '{os.getcwd()}'"
         f"\n  to   '{os.path.abspath(os.path.join(os.getcwd(),'..'))}'.\n")
    os.chdir('..')
    cwd = os.getcwd()
assert os.path.exists('./tests')
project_dir = os.path.basename(cwd)
test_dir = os.path.join(cwd,'tests')
test_data_dir = os.path.join(test_dir,'data')
# Make sure that we can import the module being tested. When running
# 'make test" and 'pytest' in the project directory, the current working
# directory is not automatically added to sys.path.
if not ('.' in sys.path or os.getcwd() in sys.path):
    echo(f"Adding '.' to sys.path.\n")
    sys.path.insert(0, '.')
#===============================================================================
from lrcmd                import run,__version__
from lrcmd.postprocessors import list_of_lines,list_of_non_empty_lines
from lrcmd.commands       import ensure_dir,exists,remove,touch, glob,rename,env
from lrcmd.exceptions     import NonZeroReturnCode, Stderr,RepeatedExecutionFailed,\
                                  CommandTimedOut
#===============================================================================    
# setup a logger which writes to stderr and to file lrcmd.log.txt
import logging
lrcmd_log = logging.getLogger('lrcmd_log')
logfile_handler = logging.FileHandler("lrcmd.log.txt",mode='w')
logfile_formatter = logging.Formatter(f"%(levelname)s: %(name)s (lrcmd v{__version__}) : %(asctime)s %(message)s\n")
logfile_handler.setFormatter(logfile_formatter)
stderr_handler = logging.StreamHandler(sys.stderr)
stderr_handler.setLevel(logging.INFO)
stderr_formatter = logging.Formatter(f"%(levelname)s: %(name)s (lrcmd v{__version__}) %(message)s\n")
stderr_handler.setFormatter(stderr_formatter)
lrcmd_log.addHandler(logfile_handler)
lrcmd_log.addHandler(stderr_handler)
#===============================================================================
import pytest
#===============================================================================
# tests
#===============================================================================
def test_echo():
    s = 'hello world'
    cmd = 'echo '+s
    result = run(cmd)
    assert result.returncode==0
    assert result.stderr==''
    assert result.stdout==s+'\n'
#===============================================================================
def test_printf_2_lines():
    s = 'hello world\nhow do you do?\n'
    cmd = f"printf '{s}'"
    result = run(cmd)
    assert result.returncode==0
    assert result.stderr==''
    assert result.stdout==s
#===============================================================================
def test_list_of_lines():
    s = 'hello world\n\nhow do you do?\n'
    cmd = f"printf '{s}'"
    result = run(cmd,post_processor=list_of_lines)
    assert result.returncode==0
    assert result.stderr==''
    expected = s.splitlines(keepends=False)
    expected.append('')
    assert result.processed==expected
#===============================================================================
def test_list_of_non_empty_lines():
    s = 'hello world\n\nhow do you do?\n'
    cmd = f"printf '{s}'"
    result = run(cmd,post_processor=list_of_non_empty_lines)
    assert result.returncode==0
    assert result.stderr==''
    expected = s.splitlines(keepends=False)
    expected = [line for line in expected if line]
    assert result.processed==expected
#===============================================================================
def test_exists_existing():
    assert exists(test_data_dir)
#===============================================================================
def test_exists_inexisting():
    assert not exists(os.path.join(test_data_dir,'inexisting'))
#===============================================================================
def test_ensure_dir_existing():
    assert exists(test_data_dir)
    ensure_dir(test_data_dir)
    assert exists(test_data_dir)
#===============================================================================
def test_ensure_dir_inexisting():
    # make sure that inexisting does not yet exists
    inexisting = os.path.join(test_data_dir,'inexisting')
    if exists(inexisting):
        remove(inexisting)
    assert not exists(inexisting)
    
    # make inexisting and check
    ensure_dir(inexisting)
    assert exists(inexisting)
    # clean up
    remove(inexisting)
    assert not exists(inexisting)    
#===============================================================================
def test_ls_inexisting():
    # ls on a inexisting directory produces a nonzero return code (1), and a
    # diagnostic message on stderr
    cmd = f"ls {os.path.join(test_data_dir,'inexisting')}"
    with pytest.raises(NonZeroReturnCode):
        try:
            run(cmd)
        except NonZeroReturnCode as e: 
            assert e.result.returncode == 1
            assert bool(e.result.stderr)
            raise
    with pytest.raises(Stderr):
        try:
            run(cmd,stderr_is_error=True,check=False)
        except Stderr as e:
            assert e.result.returncode == 1
            assert bool(e.result.stderr)
            raise
#===============================================================================
def test_no_timeout():
    # this must not timeout
    timeout = 1
    sleep = timeout/2
    cmd = f"sleep {sleep}"
    result = run(cmd,timeout=timeout)
    assert result.returncode == 0
    assert not bool(result.stdout)
    assert not bool(result.stderr)
#===============================================================================
def test_timeout():
    # this must timeout
    timeout = 1
    sleep = timeout
    cmd = f"sleep {sleep}"
    with pytest.raises(CommandTimedOut):
        run(cmd,timeout=timeout,error_log=lrcmd_log)
#===============================================================================
def test_repeated_execution_fails():
    with pytest.raises(RepeatedExecutionFailed):
        try:
            run("sleep 1",timeout=.9,attempts=3,verbose=True,wait=.5)
        except RepeatedExecutionFailed as e:
            echo('\nRaised '+type(e).__name__+':\n'+str(e),err=True)
            raise
#===============================================================================
def test_repeated_execution_succeeds():
    newdir = os.path.join(test_data_dir,'newdir')
    try:    remove(newdir)
    except: pass
    
    # this command should succeeds the second time
    cmd = "./test_succeeds_the_second_time.sh"
    result = run(cmd,working_directory=test_data_dir,attempts=3,verbose=True,wait=.5)
    assert result.attempts == 2
    assert exists(newdir)
        
    # clean up
    remove(newdir)
#===============================================================================
def test_touch():
    path = os.path.join(test_data_dir,'touch')
    if exists(path):
        remove(path)
    ensure_dir(path)
    
    for file in ['a.txt','b.txt','c.txt']:
        touch(file,path=path)
    for file in ['a.txt','b.txt','c.txt']:
        assert exists(os.path.join(path,file))
    remove(path)
#===============================================================================
def test_glob():
    path = os.path.join(test_data_dir,'glob')
    if exists(path):
        remove(path)
    ensure_dir(path)
    files = ['a.txt','b.txt','c.txt','aa.txt']
    for file in files:
        touch(file,path=path)
    dirs =['a.tx','d.txt']
    for folder in dirs:
        ensure_dir(os.path.join(path,folder))
        
    result = glob('*.txt',path=path)
    assert len(result)==4
    for file in result:
        assert file in files
         
    result = glob('?.txt',path=path)
    assert len(result)==3
    for file in result:
        assert file in files
        
    # clean up
    remove(path)
#===============================================================================
def test_rename():
    # set up
    test_dir = os.path.join(test_data_dir,'rename')
    ensure_dir(test_dir)
    old = os.path.join(test_dir,'old.txt')
    touch(old)
    assert exists(old)
    
    # test
    new = os.path.join(test_dir,'new.txt')
    rename(old,new)
    assert not exists(old)
    assert     exists(new)
    
    # clean up
    remove(test_dir)
#===============================================================================
def test_env():
    assert env('USER') == 'etijskens'
    assert env('$USER') == 'etijskens'
    assert env( 'varthatdoesnotexist')==''
    assert env('$varthatdoesnotexist')==''
#     assert env( 'varthatdoesnotexist',leibniz)==''
#     assert env('$varthatdoesnotexist',leibniz)==''
    assert env('UsR') == ''
#===============================================================================
    
#===============================================================================
# The code below is for debugging a particular test in eclipse/pydev.
# (normally all tests are run with pytest)
#===============================================================================
if __name__=="__main__":
    the_test_you_want_to_debug = test_exists_existing

    from execution_trace import trace
    with trace(f"__main__ running {the_test_you_want_to_debug}",'-*# finished #*-',singleline=False,combine=False):
        the_test_you_want_to_debug()
#===============================================================================


