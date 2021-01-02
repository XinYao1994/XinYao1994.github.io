

## Binary Numbers
## Integer and Real Numbers

## Shell
## Processes
- fork
```
pid = os.fork()
if pid == 0:
    print ("\tHi! I'm the child process.")
    print ("\tsleeping for 5 seconds....")
    time.sleep(5)
    print ("\t... exit with status 99.")
    sys.exit(99)
elif pid == -1:
    print ("yikes! fork failed!")
    sys.exit(1)
else:
    print ("I'm the parent process")
    print ("waiting for child with PID", pid)
    wval = os.wait()
    print ("wait over! process ID was ")
    print (wval[0], "exit status",wval[1]>>8)
```
## File Systems

## Memory Management

## Scheduling Processes

## Boot Sequence
- power on -> BIOS running -> MBR loaded -> GRUB loaded -> kernel loaded -> init started -> system processes running -> system operating normally

## Virtual Machines
## Containers
## Networks
## Domain Name System
## Application Layer
## Computer and Network Security
## Cloud Computing



