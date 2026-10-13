"""Windows-safe, bounded GenPRM mathematical-code service; no arbitrary exec."""
from pathlib import Path
import ctypes
from ctypes import wintypes as wt
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time

def windows_job(proc):
    """Kill worker on owner exit; cap its committed memory at 1 GiB."""
    if os.name!='nt': return None
    class BASIC(ctypes.Structure):
        _fields_=[('process_time',ctypes.c_int64),('job_time',ctypes.c_int64),('flags',wt.DWORD),
                  ('min_working_set',ctypes.c_size_t),('max_working_set',ctypes.c_size_t),
                  ('active_processes',wt.DWORD),('affinity',ctypes.c_size_t),
                  ('priority',wt.DWORD),('scheduling',wt.DWORD)]
    class IO(ctypes.Structure):
        _fields_=[(x,ctypes.c_uint64) for x in ('read_ops','write_ops','other_ops','read_bytes','write_bytes','other_bytes')]
    class EXTENDED(ctypes.Structure):
        _fields_=[('basic',BASIC),('io',IO),('process_memory',ctypes.c_size_t),('job_memory',ctypes.c_size_t),
                  ('peak_process',ctypes.c_size_t),('peak_job',ctypes.c_size_t)]
    dll=ctypes.WinDLL('kernel32',use_last_error=True)
    dll.CreateJobObjectW.argtypes=[ctypes.c_void_p,wt.LPCWSTR];dll.CreateJobObjectW.restype=wt.HANDLE
    dll.SetInformationJobObject.argtypes=[wt.HANDLE,ctypes.c_int,ctypes.c_void_p,wt.DWORD];dll.SetInformationJobObject.restype=wt.BOOL
    dll.AssignProcessToJobObject.argtypes=[wt.HANDLE,wt.HANDLE];dll.AssignProcessToJobObject.restype=wt.BOOL
    dll.CloseHandle.argtypes=[wt.HANDLE];dll.CloseHandle.restype=wt.BOOL
    handle=dll.CreateJobObjectW(None,None)
    if not handle: raise ctypes.WinError(ctypes.get_last_error())
    limits=EXTENDED();limits.basic.flags=0x2000|0x100|0x8;limits.basic.active_processes=1;limits.process_memory=1024**3
    if not dll.SetInformationJobObject(handle,9,ctypes.byref(limits),ctypes.sizeof(limits)) or not dll.AssignProcessToJobObject(handle,int(proc._handle)):
        err=ctypes.get_last_error();dll.CloseHandle(handle);raise ctypes.WinError(err)
    return dll,handle

class CodeExecutor:
    def __init__(self):
        self.proc=None;self.job=None;self.answers=None;self.history=[]

    def start(self):
        if self.proc is not None and self.proc.poll() is None: return
        self.proc=subprocess.Popen([sys.executable,'-I','-B','-u',str(Path(__file__).with_name('genprm_math_sandbox_v2.py'))],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,encoding='utf-8',
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        try:
            self.job=windows_job(self.proc);self.answers=queue.Queue();answers=self.answers;proc=self.proc
            def reader():
                for line in proc.stdout: answers.put(line)
                answers.put(None)
            threading.Thread(target=reader,daemon=True).start()
            if json.loads(self.answers.get(timeout=10))!={'ready':True}: raise ValueError('Math worker not ready')
            # Rebuild only deterministic pre-timeout state, never read host snapshots.
            for event in self.history:
                if event['status'] not in ('timeout','worker_exit'):
                    result=self.send(event['code'])
                    if result['status']!=event['status'] or result['output']!=event['output']: raise ValueError('Executor replay drift')
        except BaseException:
            self.close();raise

    def send(self,code):
        self.proc.stdin.write(json.dumps(dict(code=code),ensure_ascii=True)+'\n');self.proc.stdin.flush()
        try:
            line=self.answers.get(timeout=5)
            if line is None:
                self.close();return dict(status='worker_exit',output='Code execute Error: bounded worker exited')
            return json.loads(line)
        except queue.Empty:
            self.close();return dict(status='timeout',output='Code execute time out: 5 second limit')

    def execute(self,text):
        matches=re.findall(r'```python\s*(.*?)\s*```',text,re.DOTALL)
        if not matches: return dict(status='format_error',output='Code format error: No code found.',code=None,seconds=0.)
        self.start();start=time.monotonic();code=matches[-1].strip()
        result=dict(self.send(code),code=code,seconds=time.monotonic()-start)
        self.history.append(result)
        return result

    def reset(self):
        """Reuse the bounded child process but never carry state across items."""
        self.history=[]
        self.start()
        self.proc.stdin.write(json.dumps(dict(reset=True))+"\n");self.proc.stdin.flush()
        if json.loads(self.answers.get(timeout=5))!={"status":"ok","output":"RESET"}:
            raise ValueError("Executor reset failed")

    def replay(self,events):
        """Restore state of sealed steps. Reject discrepancies rather than reuse bad state."""
        for event in events:
            if event.get('code') is None: continue
            if event['status'] in ('timeout','worker_exit'):
                self.history.append(event);continue
            self.start();result=self.send(event['code'])
            if any(result[k]!=event[k] for k in ('status','output')): raise ValueError('Sealed code replay mismatch')
            self.history.append(event)

    def close(self):
        if self.proc is not None:
            if self.proc.poll() is None: self.proc.kill()
            self.proc.wait(timeout=3)
            for stream in (self.proc.stdin,self.proc.stdout):
                if stream: stream.close()
            self.proc=None
        if self.job is not None:
            dll,handle=self.job;dll.CloseHandle(handle);self.job=None

    def __enter__(self): return self
    def __exit__(self,*args): self.close()
